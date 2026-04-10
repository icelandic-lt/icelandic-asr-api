"""Verify ASR transcriptions against reference texts.

Sends each test-audio/*.wav to the server and compares the result
against the corresponding .txt file using character-level similarity.
Tests both auto language detection and explicit language=is.
"""

import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

MIN_SIMILARITY = 0.95  # 95% character similarity required


def normalize(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def similarity(a, b):
    """Character-level similarity ratio (0..1)."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def transcribe(url, wav_path, language=None):
    """Send a WAV file to the server, return text."""
    import mimetypes

    boundary = "----TestBoundary12345"
    body = b""

    # file field
    body += f"--{boundary}\r\n".encode()
    body += (
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{wav_path.name}"\r\n'
    ).encode()
    mime = mimetypes.guess_type(str(wav_path))[0] or "audio/wav"
    body += f"Content-Type: {mime}\r\n\r\n".encode()
    body += wav_path.read_bytes()
    body += b"\r\n"

    # language field (optional)
    if language:
        body += f"--{boundary}\r\n".encode()
        body += (
            'Content-Disposition: form-data; name="language"\r\n' "\r\n"
        ).encode()
        body += f"{language}\r\n".encode()

    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["text"]


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    url = f"{base_url}/v1/audio/transcriptions"
    test_dir = Path(__file__).parent.parent / "test-audio"

    wavs = sorted(test_dir.glob("*.wav"))
    if not wavs:
        print(f"ERROR: no WAV files in {test_dir}")
        sys.exit(1)

    failures = 0
    total = 0

    for wav in wavs:
        ref_file = wav.with_suffix(".txt")
        if not ref_file.exists():
            print(f"SKIP {wav.name}: no reference .txt")
            continue

        ref = normalize(ref_file.read_text(encoding="utf-8"))

        for mode, lang in [("auto", None), ("is", "is")]:
            total += 1
            asr = normalize(transcribe(url, wav, language=lang))
            sim = similarity(ref, asr)
            status = "PASS" if sim >= MIN_SIMILARITY else "FAIL"

            print(f"{status} {wav.stem} ({mode}) sim={sim:.0%}")
            print(f"  REF: {ref}")
            print(f"  ASR: {asr}")

            if status == "FAIL":
                failures += 1

    print(
        f"\n{total - failures}/{total} passed "
        f"(threshold: {MIN_SIMILARITY:.0%})"
    )

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
