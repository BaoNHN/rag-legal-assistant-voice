"""
voice/voice_engine.py
Orchestrates the full voice pipeline:
  Audio → STT (Whisper) → [existing RAG engine] → TTS (edge-TTS) → RVC → Audio

This file is the only integration point with the existing codebase.
It imports from engine/rag_engine.py but does NOT modify it.
"""

import time
from voice import stt, tts, rvc


def transcribe_audio(audio_bytes: bytes, mime: str = "audio/webm") -> dict:
    """
    Stage 1–2: Receive audio, return transcription.
    Called by the /voice/transcribe Flask route.
    """
    t0 = time.perf_counter()
    result = stt.transcribe(audio_bytes, mime)
    result["latency_stt_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


def speak_answer(answer_text: str, voice: str = None, use_rvc: bool = True) -> dict:
    """
    Stages 4–6: Convert answer text to personalized spoken audio.
    Called by the /voice/speak Flask route.

    Parameters
    ----------
    answer_text : str   Full answer from the RAG pipeline
    voice       : str   edge-TTS voice override (optional)
    use_rvc     : bool  Whether to apply RVC conversion

    Returns
    -------
    dict {
        "audio":        bytes  — final WAV/MP3 audio,
        "mime":         str    — audio MIME type,
        "rvc_used":     bool,
        "latency_tts_ms": int,
        "latency_rvc_ms": int,
    }
    """
    # Truncate long answers for the spoken path (Thesis Technique T4)
    spoken_text = tts.truncate_for_speech(answer_text, max_words=150)

    # Stage 4 — TTS
    t0 = time.perf_counter()
    tts_audio = tts.synthesize(spoken_text, voice=voice or tts.DEFAULT_VOICE)
    latency_tts = round((time.perf_counter() - t0) * 1000)

    # Stage 5 — RVC (optional, graceful fallback if Colab tunnel is down)
    rvc_used = False
    t1 = time.perf_counter()
    if use_rvc and rvc.RVC_ENDPOINT:
        final_audio = rvc.convert(tts_audio, mime="audio/mp3")
        rvc_used = final_audio is not tts_audio
    else:
        final_audio = tts_audio
    latency_rvc = round((time.perf_counter() - t1) * 1000)

    mime = "audio/wav" if rvc_used else "audio/mpeg"

    return {
        "audio":          final_audio,
        "mime":           mime,
        "rvc_used":       rvc_used,
        "latency_tts_ms": latency_tts,
        "latency_rvc_ms": latency_rvc,
    }
