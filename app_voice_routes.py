"""
app_voice_routes.py
─────────────────────────────────────────────────────────────────────────────
PASTE these two routes into your existing app.py.

Step 1 — Add this import block near the top of app.py (after existing imports):
────────────────────────────────────────────────────────────────────────────
from voice.voice_engine import transcribe_audio, speak_answer

Step 2 — Paste the two route functions below anywhere before  if __name__ == "__main__":
─────────────────────────────────────────────────────────────────────────────
"""

# ── Route 1 ──────────────────────────────────────────────────────────────────
@app.route("/voice/transcribe", methods=["POST"])
def voice_transcribe():
    """
    Receive audio blob from browser MediaRecorder, return JSON transcription.
    Frontend posts:  multipart/form-data  field name = "audio"
    """
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401

    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    mime = audio_file.content_type or "audio/webm"

    import time
    t_start = time.perf_counter()
    result = transcribe_audio(audio_bytes, mime)
    result["latency_total_ms"] = round((time.perf_counter() - t_start) * 1000)

    return jsonify(result)


# ── Route 2 ──────────────────────────────────────────────────────────────────
@app.route("/voice/speak", methods=["POST"])
def voice_speak():
    """
    Convert answer text to personalized spoken audio (TTS + RVC).
    Frontend posts JSON:  { "text": "...", "voice": "vi-VN-HoaiMyNeural" }
    Returns raw audio bytes with correct Content-Type.
    """
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401

    data        = request.get_json(force=True)
    answer_text = data.get("text", "").strip()
    voice       = data.get("voice", None)   # optional override
    use_rvc     = data.get("use_rvc", True)

    if not answer_text:
        return jsonify({"error": "no text provided"}), 400

    result = speak_answer(answer_text, voice=voice, use_rvc=use_rvc)

    response = app.response_class(
        response=result["audio"],
        status=200,
        mimetype=result["mime"],
    )
    # Expose latency metrics in headers for evaluation (Thesis Section 6.2.2)
    response.headers["X-Latency-TTS-ms"] = result["latency_tts_ms"]
    response.headers["X-Latency-RVC-ms"] = result["latency_rvc_ms"]
    response.headers["X-RVC-Used"]        = str(result["rvc_used"]).lower()
    return response


# ── Route 3 (health check) ────────────────────────────────────────────────────
@app.route("/voice/status")
def voice_status():
    """Returns JSON showing which voice components are active."""
    from voice import rvc, tts
    return jsonify({
        "whisper_model": __import__("os").getenv("WHISPER_MODEL", "small"),
        "tts_voice":     tts.DEFAULT_VOICE,
        "rvc_endpoint":  rvc.RVC_ENDPOINT or None,
        "rvc_available": rvc.is_available(),
    })
