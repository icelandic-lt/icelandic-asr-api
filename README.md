# Icelandic ASR Service

This repository provides an **OpenAI Whisper-compatible ASR service** with automatic language detection and optimized Icelandic speech-to-text.

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
[![CI Status](https://github.com/icelandic-lt/icelandic-asr-api/actions/workflows/ci.yml/badge.svg)](https://github.com/icelandic-lt/icelandic-asr-api/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Overview

**Icelandic ASR Service** has been created by [Grammatek ehf](https://www.grammatek.com) and is part of the [Icelandic Language Technology Programme](https://github.com/icelandic-lt/icelandic-lt).

- **Category:** [ASR](https://github.com/icelandic-lt/icelandic-lt/blob/main/doc/asr.md)
- **Domain:** Server / API
- **Languages:** Python
- **Language Version/Dialect:**
  - Python: 3.11 - 3.13
- **Audience:** Developers, Researchers
- **Origins:** Icelandic Language and Voice Lab models

## Status
![Development](https://img.shields.io/badge/Development-yellow)

## System Requirements
- **Operating System:** Linux (recommended), macOS, Windows
- **Python:** 3.11+
- **GPU:** CUDA-compatible GPU(s) recommended (CPU supported but slower)
- **Models:** Downloaded automatically from HuggingFace on first startup

## Description

This server provides an OpenAI Whisper-compatible API (`POST /v1/audio/transcriptions`) with automatic language detection and transcription for all Whisper-supported languages. For Icelandic audio, dedicated models are used for better accuracy.

[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) is always loaded for automatic language detection and multilingual transcription. When Icelandic is detected (or explicitly requested via the `language` parameter), a dedicated Icelandic model handles the transcription:

| `--is-model`        | Backend         | HuggingFace Model | Notes |
|---------------------|-----------------|-------|-------|
| `wav2vec2` (default)| wav2vec2 CTC    | [wav2vec2-large-xlsr-53-icelandic-ep30-967h](https://huggingface.co/language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h) ([model card](https://huggingface.co/language-and-voice-lab/wav2vec2-large-xlsr-53-icelandic-ep30-967h)) | Fastest, recommended for real-time use |
| `whisper-icelandic` | faster-whisper  | [whisper-large-icelandic-62640-steps-967h-ct2](https://huggingface.co/language-and-voice-lab/whisper-large-icelandic-62640-steps-967h-ct2) ([model card](https://huggingface.co/language-and-voice-lab/whisper-large-icelandic-62640-steps-967h-ct2)) | Better quality, but significantly slower |
| `none`              | —               | — | Turbo only, no dedicated Icelandic model |

The Icelandic models are from the [Language and Voice Lab](https://huggingface.co/language-and-voice-lab), trained on 967h of Icelandic speech data.

**Multi-GPU:** By default (`--devices auto`), the server loads one model instance per available GPU and dispatches requests round-robin across all workers. With *N* GPUs, *N* requests can run in parallel. On a single GPU (or CPU), requests are serialized automatically. Models are downloaded from HuggingFace on first startup.

**Dataset verification:** This service can be used together with [Revoxx](https://github.com/icelandic-lt/revoxx) to verify speech dataset quality. Revoxx supports ASR-based recording verification via any OpenAI-compatible ASR endpoint — point it at this server to automatically transcribe recordings and compare them against the script text.

## Installation

<details>
<summary><b>Basic Installation</b></summary>

### Using uv

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
uv pip install icelandic-asr
```

### Using pip

```bash
pip install icelandic-asr
```

### From source

```bash
git clone https://github.com/icelandic-lt/icelandic-asr-api.git
cd icelandic-asr-api
uv pip install .    # or: pip install .
```

</details>

<details>
<summary><b>Development Setup</b></summary>

```bash
git clone https://github.com/icelandic-lt/icelandic-asr-api.git
cd icelandic-asr-api

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Running code quality checks

```bash
# Format code
black icelandic_asr/

# Check code style
flake8 icelandic_asr/
```

### Running transcription tests

With a running server, verify transcription accuracy against the included Icelandic test audio:

```bash
# Against local server (default: http://localhost:8000)
python tests/verify_transcriptions.py

# Against a custom endpoint
python tests/verify_transcriptions.py http://localhost:9000
```

The test sends each `test-audio/*.wav` with both auto language detection and explicit `language=is`, then compares results against the reference `.txt` files using character-level similarity (95% threshold).

</details>

<details>
<summary><b>Docker</b></summary>

The Docker image uses CUDA 12.4, compatible with Ubuntu 24.04 LTS (nvidia-driver-550).

```bash
# Build
docker build -t icelandic-asr .

# Run with GPU (share host HuggingFace cache)
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/app/models \
  icelandic-asr

# Run CPU-only
docker run -p 8000:8000 \
  -v ~/.cache/huggingface:/app/models \
  icelandic-asr --devices cpu

# Custom options
docker run --gpus all -p 9000:9000 \
  -v ~/.cache/huggingface:/app/models \
  icelandic-asr --port 9000 --is-model whisper-icelandic
```

Models are downloaded on first start and cached via the bind-mounted HuggingFace cache directory. This shares models between the container and the host, so they are only downloaded once.

</details>

## Running

### After installation

```bash
icelandic-asr
```

### During development

```bash
python -m icelandic_asr.server
```

### Command-line arguments

```bash
icelandic-asr --help

# Default: wav2vec2, all available GPUs
icelandic-asr

# Use faster-whisper Icelandic model (better quality)
icelandic-asr --is-model whisper-icelandic

# Turbo only (no dedicated Icelandic model)
icelandic-asr --is-model none

# Use specific GPUs
icelandic-asr --devices cuda:0,cuda:2

# Single GPU
icelandic-asr --devices cuda:0

# CPU-only
icelandic-asr --devices cpu

# Custom host/port
icelandic-asr --host 0.0.0.0 --port 9000

# Limit upload size (default: 200 MB)
icelandic-asr --max-upload-mb 500
```

## API

### `POST /v1/audio/transcriptions`

OpenAI Whisper-compatible endpoint.

**Request** (multipart form data):

| Field             | Type   | Default  | Description |
|-------------------|--------|----------|-------------|
| `file`            | binary | required | Audio file (WAV, MP3, FLAC, etc.) |
| `language`        | string | auto     | ISO-639-1 language code (e.g. `is`, `en`). Auto-detects if omitted. |
| `response_format` | string | `json`  | `json` or `text` |
| `model`           | string | —        | Ignored (for OpenAI compatibility) |

**Response** (`json` format):

```json
{"text": "transcribed text here"}
```

**Example with curl:**

```bash
# Auto-detect language
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav

# Force Icelandic (skip language detection)
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F language=is
```

**Example with OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1", file=f
    )
print(result.text)
```

Upload size is limited to 200 MB by default (configurable via `--max-upload-mb`). Uploads exceeding the limit are rejected with HTTP 413.

### `GET /v1/status`

Returns server configuration info (devices, backend, worker count, available workers).

### `GET /health`

Health check endpoint for load balancers and container orchestration. Returns `{"status": "ok"}` with HTTP 200 when the server is ready, or HTTP 503 while models are still loading.

## Acknowledgements

This project is part of the program Language Technology for Icelandic. The program was funded by the Icelandic Ministry of Culture and Business Affairs.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
