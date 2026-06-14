"""
voice/tts.py
Text-to-Speech using Microsoft edge-tts (Vietnamese neural voices).
Produces a base WAV/MP3 that is then passed to RVC for voice conversion.
"""

import asyncio
import io
import os
import tempfile

import edge_tts

# Default Vietnamese voices — change via env var
VOICE_VI_FEMALE = os.getenv("TTS_VOICE_FEMALE", "vi-VN-HoaiMyNeural")
VOICE_VI_MALE   = os.getenv("TTS_VOICE_MALE",   "vi-VN-NamMinhNeural")
DEFAULT_VOICE   = os.getenv("TTS_VOICE", VOICE_VI_FEMALE)


async def _synthesize(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def synthesize(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """
    Convert text to speech using edge-TTS.

    Parameters
    ----------
    text  : str   Answer text to speak (Vietnamese or English)
    voice : str   edge-TTS voice name

    Returns
    -------
    bytes   Raw MP3 audio bytes
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp_path = f.name

    try:
        # Use explicit loop to avoid ProactorEventLoop issues on Windows
        # when called from Flask handler threads (asyncio.run creates a new
        # loop per thread which is safe, but set_event_loop ensures cleanup).
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_synthesize(text, voice, tmp_path))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def truncate_for_speech(text: str, max_words: int = 150) -> str:
    """
    Truncate long legal answers to ~150 words for spoken output.
    Full text is still returned in the chat transcript.
    This matches Technique T4 described in the thesis (Section 6.3).
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return truncated + " … (xem toàn bộ câu trả lời trong khung chat)"
