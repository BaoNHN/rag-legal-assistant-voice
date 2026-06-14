"""
voice/rvc.py
RVC v2 voice-conversion client.
The actual RVC inference runs on Google Colab (GPU) and is exposed via a
cloudflared tunnel.  This module is the thin HTTP client that calls it.

Set the tunnel URL in your environment:
    RVC_ENDPOINT=https://your-colab-tunnel.trycloudflare.com

If the endpoint is unreachable the module falls back to returning the
original TTS audio unchanged (graceful degradation — Technique T2).
"""

import os
import requests

RVC_ENDPOINT  = os.getenv("RVC_ENDPOINT", "").rstrip("/")
RVC_TIMEOUT   = int(os.getenv("RVC_TIMEOUT", "20"))      # seconds
RVC_PITCH     = int(os.getenv("RVC_PITCH",   "0"))        # semitones
RVC_INDEX_RATE = float(os.getenv("RVC_INDEX_RATE", "0.75"))


def convert(audio_bytes: bytes, mime: str = "audio/mp3") -> bytes:
    """
    Send TTS audio to the Colab RVC server and return converted audio.

    Parameters
    ----------
    audio_bytes : bytes   Base TTS audio (MP3 or WAV from edge-TTS)
    mime        : str     MIME type of the audio

    Returns
    -------
    bytes  Converted WAV audio (or original audio if RVC is unavailable)
    """
    if not RVC_ENDPOINT:
        # RVC not configured — return TTS audio as-is
        return audio_bytes

    try:
        resp = requests.post(
            f"{RVC_ENDPOINT}/convert",
            files={"audio": ("input.mp3", audio_bytes, mime)},
            data={
                "pitch":      RVC_PITCH,
                "index_rate": RVC_INDEX_RATE,
            },
            timeout=RVC_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.content
        else:
            print(f"[RVC] Server returned {resp.status_code} — falling back to TTS audio.")
            return audio_bytes

    except requests.exceptions.RequestException as e:
        print(f"[RVC] Endpoint unreachable ({e}) — falling back to TTS audio.")
        return audio_bytes


def is_available() -> bool:
    """Check if the RVC endpoint is reachable."""
    if not RVC_ENDPOINT:
        return False
    try:
        r = requests.get(f"{RVC_ENDPOINT}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False
