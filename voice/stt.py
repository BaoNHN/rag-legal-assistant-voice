"""
voice/stt.py
Automatic Speech Recognition using OpenAI Whisper.
Drop-in for rag-legal-assistant-master — no changes to existing RAG pipeline.
"""

import os
import tempfile
import whisper

_model = None
_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")   # tiny | small | medium


def _load_model():
    global _model
    if _model is None:
        print(f"[STT] Loading Whisper-{_MODEL_SIZE} …")
        _model = whisper.load_model(_MODEL_SIZE)
        print("[STT] Whisper ready.")
    return _model


def transcribe(audio_bytes: bytes, mime: str = "audio/webm") -> dict:
    """
    Transcribe raw audio bytes to text.

    Parameters
    ----------
    audio_bytes : bytes   Raw audio from browser MediaRecorder (webm/wav/ogg)
    mime        : str     MIME type hint (unused by Whisper, kept for future use)

    Returns
    -------
    dict  {"text": str, "language": str, "segments": list}
    """
    model = _load_model()

    # Write to a temp file — Whisper needs a file path
    suffix = ".webm" if "webm" in mime else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = model.transcribe(
            tmp_path,
            language=None,          # auto-detect; set "vi" to force Vietnamese
            task="transcribe",
            fp16=False,             # CPU-safe
        )
        return {
            "text":     result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
        }
    finally:
        os.unlink(tmp_path)
