"""Icelandic ASR Service — OpenAI Whisper-compatible.

Endpoints:
    POST /v1/audio/transcriptions  — transcribe audio
    GET  /v1/status                — server info
    GET  /health                   — health check

Architecture:
    By default, each GPU gets a full worker with both Whisper turbo
    (language detection + multilingual fallback) and a dedicated
    Icelandic model.  Requests are dispatched round-robin across
    workers.  Use --turbo-device to load turbo on a single GPU only
    (saves VRAM on multi-GPU setups at the cost of serializing
    language detection).
"""

import argparse
import asyncio
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio
import uvicorn
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from omnivad import OmniVAD
import soundfile as sf
from starlette.responses import PlainTextResponse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import pipeline as hf_pipeline

log = logging.getLogger("icelandic-asr")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TURBO_MODEL_ID = "openai/whisper-large-v3-turbo"
LANG_DETECT_SECONDS = 10
DEFAULT_MAX_UPLOAD_MB = 200

IS_MODELS = {
    "wav2vec2": {
        "backend": "wav2vec2",
        "model_id": (
            "language-and-voice-lab/"
            "wav2vec2-large-xlsr-53-icelandic-ep30-967h"
        ),
    },
    "whisper-icelandic": {
        "backend": "faster-whisper",
        "model_id": (
            "language-and-voice-lab/"
            "whisper-large-icelandic-62640-steps-967h-ct2"
        ),
    },
    "none": {
        "backend": "none",
        "model_id": "",
    },
}

ASR_RESPONSE_FORMATS = {"json", "text"}

# Whisper-supported language codes (ISO-639-1).
ASR_LANGUAGES = frozenset(
    {
        "af",
        "am",
        "ar",
        "as",
        "az",
        "ba",
        "be",
        "bg",
        "bn",
        "bo",
        "br",
        "bs",
        "ca",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "eu",
        "fa",
        "fi",
        "fo",
        "fr",
        "gl",
        "gu",
        "ha",
        "haw",
        "he",
        "hi",
        "hr",
        "ht",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jw",
        "ka",
        "kk",
        "km",
        "kn",
        "ko",
        "la",
        "lb",
        "ln",
        "lo",
        "lt",
        "lv",
        "mg",
        "mi",
        "mk",
        "ml",
        "mn",
        "mr",
        "ms",
        "mt",
        "my",
        "ne",
        "nl",
        "nn",
        "no",
        "oc",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sa",
        "sd",
        "si",
        "sk",
        "sl",
        "sn",
        "so",
        "sq",
        "sr",
        "su",
        "sv",
        "sw",
        "ta",
        "te",
        "tg",
        "th",
        "tk",
        "tl",
        "tr",
        "tt",
        "uk",
        "ur",
        "uz",
        "vi",
        "yi",
        "yo",
        "yue",
        "zh",
    }
)


# ---------------------------------------------------------------------------
# Worker: one complete pipeline per device
# ---------------------------------------------------------------------------


@dataclass
class Worker:
    device: str
    turbo: object = None  # HF pipeline (or None if shared)
    is_backend: str = "none"
    is_model: object = None
    is_processor: object = None  # wav2vec2 only
    vad: OmniVAD | None = None  # for wav2vec2 chunking


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    devices: list[str] = field(default_factory=list)
    is_backend: str = "none"
    turbo_device: str | None = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_MB * 1024 * 1024
    workers: list[Worker] = field(default_factory=list)
    pool: asyncio.Queue | None = None
    ready: bool = False
    shared_turbo: object = None
    shared_turbo_lock: asyncio.Lock | None = None


state = ServerState()


# ---------------------------------------------------------------------------
# Language detection via Whisper turbo
# ---------------------------------------------------------------------------


def _detect_language(turbo, waveform, sr):
    """Detect language from the first 10s of audio.

    Returns (language_code, probability).
    """
    turbo_model = turbo.model
    turbo_processor = turbo.feature_extractor
    turbo_device = turbo_model.device

    # Only use the first N seconds for detection
    max_samples = LANG_DETECT_SECONDS * sr
    detect_wf = waveform[:max_samples]

    audio_np = detect_wf.cpu().numpy()
    if sr != 16000:
        audio_np = (
            torchaudio.functional.resample(detect_wf, sr, 16000).cpu().numpy()
        )
    features = turbo_processor(
        audio_np, sampling_rate=16000, return_tensors="pt"
    )
    input_features = features.input_features.to(
        device=turbo_device, dtype=turbo_model.dtype
    )

    with torch.no_grad():
        encoder_outputs = turbo_model.get_encoder()(input_features)

    lang_to_id = turbo_model.generation_config.lang_to_id
    id_to_lang = {v: k for k, v in lang_to_id.items()}
    lang_token_ids_list = sorted(lang_to_id.values())

    with torch.no_grad():
        decoder_input_ids = torch.tensor(
            [[turbo_model.config.decoder_start_token_id]],
            device=turbo_device,
        )
        logits = turbo_model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits[0, -1]
        lang_indices = torch.tensor(lang_token_ids_list, device=turbo_device)
        lang_logits = logits[lang_indices].float()
        lang_probs = torch.softmax(lang_logits, dim=-1)
        best_idx = torch.argmax(lang_probs).item()

    lang_token_id = lang_token_ids_list[best_idx]
    lang = id_to_lang.get(lang_token_id, "<|en|>").strip("<>|")
    prob = lang_probs[best_idx].item()
    return lang, prob


def _transcribe_turbo(turbo, waveform, sr, language):
    """Transcribe with Whisper turbo HF pipeline (handles chunking)."""
    audio_np = waveform.cpu().numpy().astype(np.float32)
    if sr != 16000:
        audio_np = (
            torchaudio.functional.resample(waveform, sr, 16000)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        sr = 16000
    result = turbo(
        {"raw": audio_np, "sampling_rate": sr},
        generate_kwargs={
            "language": language,
            "task": "transcribe",
        },
        chunk_length_s=30,
        batch_size=1,
    )
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Icelandic transcription
# ---------------------------------------------------------------------------


def _transcribe_icelandic(worker, waveform, sr):
    """Transcribe with a dedicated Icelandic model."""
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    if worker.is_backend == "faster-whisper":
        audio_np = waveform.cpu().numpy().astype(np.float32)
        segments, _ = worker.is_model.transcribe(
            audio_np,
            language="is",
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    elif worker.is_backend == "wav2vec2":
        model_device = next(worker.is_model.parameters()).device
        mono_np = waveform.cpu().numpy()

        # Use OmniVAD to find speech segments
        result = worker.vad.detect(mono_np)
        timestamps = result.get("timestamps", [])

        if not timestamps:
            # No speech detected — transcribe entire audio
            timestamps = [(0, len(mono_np) / 16000)]

        texts = []
        for start_sec, end_sec in timestamps:
            s = int(start_sec * 16000)
            e = int(end_sec * 16000)
            chunk = waveform[s:e]
            if chunk.shape[0] == 0:
                continue
            inputs = worker.is_processor(
                chunk.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            iv = inputs.input_values.to(model_device)
            with torch.no_grad():
                logits = worker.is_model(iv).logits
            ids = torch.argmax(logits, dim=-1)
            t = worker.is_processor.batch_decode(ids)[0].strip()
            if t:
                texts.append(t)
        return " ".join(texts)


# ---------------------------------------------------------------------------
# Full transcription pipeline (runs on a single worker)
# ---------------------------------------------------------------------------


def _transcribe(worker, waveform, sr, language=None):
    """Full pipeline: detect language → route → transcribe."""
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    audio_duration = round(waveform.shape[-1] / sr, 3)

    t0 = time.perf_counter()
    turbo = worker.turbo or state.shared_turbo

    # Language detection (or use provided language)
    if language:
        lang = language
        lang_prob = 1.0
        lang_detect_time = 0
    else:
        t_lang = time.perf_counter()
        lang, lang_prob = _detect_language(turbo, waveform, sr)
        lang_detect_time = time.perf_counter() - t_lang

    # Route: Icelandic → dedicated model, else → turbo
    use_is = (
        lang == "is"
        and worker.is_backend != "none"
        and worker.is_model is not None
    )

    if use_is:
        t_is = time.perf_counter()
        text = _transcribe_icelandic(worker, waveform, sr)
        is_time = time.perf_counter() - t_is
        asr_time = time.perf_counter() - t0
        log.info(
            f"ASR [{worker.device}] lang={lang} "
            f"(p={lang_prob:.2f}) "
            f"transcribed {audio_duration}s in "
            f"{asr_time:.3f}s "
            f"(detect={lang_detect_time:.3f}s, "
            f"{worker.is_backend}={is_time:.3f}s, "
            f"RTF={asr_time / audio_duration:.2f})"
            f" → {text[:80]!r}"
        )
    else:
        t_decode = time.perf_counter()
        text = _transcribe_turbo(turbo, waveform, sr, lang)
        decode_time = time.perf_counter() - t_decode
        asr_time = time.perf_counter() - t0
        log.info(
            f"ASR [{worker.device}] lang={lang} "
            f"(p={lang_prob:.2f}) "
            f"transcribed {audio_duration}s in "
            f"{asr_time:.3f}s "
            f"(detect={lang_detect_time:.3f}s, "
            f"decode={decode_time:.3f}s, "
            f"RTF={asr_time / audio_duration:.2f})"
            f" → {text[:80]!r}"
        )

    return text


# ---------------------------------------------------------------------------
# Worker loading
# ---------------------------------------------------------------------------


def _load_turbo(device):
    """Load Whisper turbo on *device*."""
    asr_dtype = torch.float16 if "cuda" in device else torch.float32
    log.info(f"Loading Whisper turbo on {device}: " f"{TURBO_MODEL_ID}...")
    return hf_pipeline(
        "automatic-speech-recognition",
        model=TURBO_MODEL_ID,
        dtype=asr_dtype,
        device=device,
    )


def _load_worker(device, is_backend, load_turbo=True):
    """Load a full worker on *device*."""
    worker = Worker(device=device, is_backend=is_backend)

    if load_turbo:
        worker.turbo = _load_turbo(device)

    if is_backend == "none":
        return worker

    preset_key = {
        "faster-whisper": "whisper-icelandic",
        "wav2vec2": "wav2vec2",
    }[is_backend]
    preset = IS_MODELS[preset_key]

    if is_backend == "faster-whisper":
        use_cuda = "cuda" in device
        di = int(device.split(":")[-1]) if use_cuda else 0
        ct = "int8_float16" if use_cuda else "int8"
        log.info(
            f"Loading faster-whisper on {device} ({ct}): "
            f"{preset['model_id']}..."
        )
        worker.is_model = WhisperModel(
            preset["model_id"],
            device="cuda" if use_cuda else "cpu",
            device_index=di,
            compute_type=ct,
        )

    elif is_backend == "wav2vec2":
        log.info(f"Loading wav2vec2 on {device}: " f"{preset['model_id']}...")
        worker.is_processor = Wav2Vec2Processor.from_pretrained(
            preset["model_id"]
        )
        worker.is_model = Wav2Vec2ForCTC.from_pretrained(
            preset["model_id"]
        ).to(device)
        worker.is_model.eval()
        log.info("Loading OmniVAD for wav2vec2 segmentation...")
        worker.vad = OmniVAD()

    return worker


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.pool = asyncio.Queue()
    shared_turbo_mode = state.turbo_device is not None

    # Shared turbo mode: load turbo once, workers without
    if shared_turbo_mode:
        state.shared_turbo = _load_turbo(state.turbo_device)
        state.shared_turbo_lock = asyncio.Lock()

    for device in state.devices:
        worker = _load_worker(
            device,
            state.is_backend,
            load_turbo=not shared_turbo_mode,
        )
        state.workers.append(worker)
        state.pool.put_nowait(worker)

    turbo_info = (
        f"shared on {state.turbo_device}"
        if shared_turbo_mode
        else "per-worker"
    )
    is_info = state.is_backend if state.is_backend != "none" else "turbo-only"
    log.info(
        f"ASR ready — turbo: {turbo_info}, "
        f"Icelandic: {is_info}, "
        f"workers: {[w.device for w in state.workers]}"
    )
    state.ready = True
    yield
    state.ready = False


app = FastAPI(title="Icelandic ASR Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions (OpenAI Whisper-compatible)
# ---------------------------------------------------------------------------


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    req_id = uuid.uuid4().hex[:8]
    form = await request.form()
    file = form.get("file")
    if file is None:
        raise HTTPException(400, "Missing 'file' field")

    response_format = (form.get("response_format") or "json").strip().lower()
    if response_format not in ASR_RESPONSE_FORMATS:
        raise HTTPException(
            400,
            f"Invalid 'response_format': "
            f"{response_format!r}. "
            f"Supported: {sorted(ASR_RESPONSE_FORMATS)}",
        )

    language = form.get("language")
    if language:
        language = language.strip().lower() or None
        if language and language not in ASR_LANGUAGES:
            raise HTTPException(
                400,
                f"Invalid 'language': {language!r}. "
                f"Must be a Whisper-supported "
                f"ISO-639-1 code.",
            )

    audio_content = await file.read()
    upload_mb = len(audio_content) / (1024 * 1024)

    if len(audio_content) > state.max_upload_bytes:
        max_mb = state.max_upload_bytes / (1024 * 1024)
        raise HTTPException(
            413,
            f"Upload too large: {upload_mb:.1f} MB " f"(max {max_mb:.0f} MB)",
        )

    log.info(
        f"[{req_id}] upload={upload_mb:.1f}MB" f" lang={language or 'auto'}"
    )

    buf = io.BytesIO(audio_content)
    try:
        data, sr = sf.read(buf, dtype="float32")
        waveform = torch.from_numpy(data)
        if waveform.dim() == 2:
            # (samples, channels) → (channels, samples)
            waveform = waveform.T
    except Exception as e:
        log.warning(f"[{req_id}] decode failed: {e}")
        raise HTTPException(400, f"Could not decode audio file: {e}")

    # Grab a worker (round-robin)
    worker = await state.pool.get()
    try:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            _transcribe,
            worker,
            waveform,
            sr,
            language,
        )
    except Exception:
        log.exception(f"[{req_id}] transcription failed")
        raise HTTPException(500, "Transcription failed")
    finally:
        state.pool.put_nowait(worker)

    if response_format == "text":
        return PlainTextResponse(text)

    return {"text": text}


# ---------------------------------------------------------------------------
# GET /v1/status
# ---------------------------------------------------------------------------


@app.get("/v1/status")
async def server_status():
    turbo_mode = (
        f"shared:{state.turbo_device}" if state.turbo_device else "per-worker"
    )
    return {
        "devices": [w.device for w in state.workers],
        "turbo": turbo_mode,
        "is_backend": state.is_backend,
        "workers": len(state.workers),
        "available": (state.pool.qsize() if state.pool else 0),
    }


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    if not state.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading"},
        )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Icelandic ASR Service")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--devices",
        default="auto",
        help=(
            "Comma-separated CUDA devices or 'auto' for "
            "all GPUs (default: auto). "
            "Examples: cuda:0  cuda:0,cuda:1  auto  cpu"
        ),
    )
    p.add_argument(
        "--is-model",
        default="wav2vec2",
        choices=list(IS_MODELS.keys()),
        help=(
            "Icelandic ASR model (default: wav2vec2). "
            "Whisper turbo is always loaded for language "
            "detection and non-Icelandic transcription. "
            "wav2vec2 (CTC, fastest), "
            "whisper-icelandic (faster-whisper, better "
            "quality), "
            "none (turbo only, no dedicated IS model)"
        ),
    )
    p.add_argument(
        "--max-upload-mb",
        type=int,
        default=DEFAULT_MAX_UPLOAD_MB,
        help=(
            f"Maximum upload size in MB " f"(default: {DEFAULT_MAX_UPLOAD_MB})"
        ),
    )
    p.add_argument(
        "--turbo-device",
        default=None,
        help=(
            "Load Whisper turbo on this device only "
            "instead of on every GPU (saves VRAM). "
            "Language detection is then serialized "
            "through this single device. "
            "Default: turbo on every GPU."
        ),
    )
    args = p.parse_args()

    log_fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    # Apply same format to uvicorn loggers
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_log = logging.getLogger(name)
        uv_log.handlers.clear()
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(log_fmt))
        uv_log.addHandler(h)

    # Resolve devices
    if args.devices == "auto":
        if torch.cuda.is_available():
            state.devices = [
                f"cuda:{i}" for i in range(torch.cuda.device_count())
            ]
        else:
            state.devices = ["cpu"]
    else:
        state.devices = [d.strip() for d in args.devices.split(",")]

    state.is_backend = IS_MODELS[args.is_model]["backend"]
    state.turbo_device = args.turbo_device
    state.max_upload_bytes = args.max_upload_mb * 1024 * 1024

    log.info(
        f"Devices: {state.devices}, "
        f"IS backend: {state.is_backend}, "
        f"turbo: {'shared:' + state.turbo_device if state.turbo_device else 'per-worker'}"  # noqa: E501
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
