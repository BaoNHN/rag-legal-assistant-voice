from flask import Flask, render_template, request, session, redirect, jsonify
import os
import uuid
import threading

from engine.rag_engine import ask_rag
from database.database import (
    init_db, get_conn,
    login_user,
    create_chat, get_all_chats,
    save_message, get_messages,
    rename_chat, delete_chat, upsert_import_chat
)
from engine.import_law_engine import run_import, get_job

# Voice deps are optional — install requirements_voice.txt to enable
_VOICE_ERROR = None
try:
    from voice.voice_engine import transcribe_audio, speak_answer
except ImportError as _e:
    _VOICE_ERROR = str(_e)
    transcribe_audio = speak_answer = None
# Note: single users table (role 0=student, 1=teacher)

app = Flask(__name__)
app.secret_key = "secret_key"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads_tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

init_db()


# ── Helpers ──────────────────────────────────────────────────────────────────
def logged_in():
    return "user_id" in session

def is_teacher():
    return session.get("user_type") == "teacher"


# ── Pages ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    if not logged_in():
        return render_template("login.html")
    return render_template("index.html", is_teacher=is_teacher())


@app.route("/import")
def import_page():
    if not logged_in() or not is_teacher():
        return redirect("/")
    return render_template("import_law.html")


# ── Auth ─────────────────────────────────────────────────────────────────────
@app.route("/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("student_name") or data.get("username", "")
    password = data.get("password", "")

    user = login_user(username, password)
    if user:
        session["user_id"]   = user["user_id"]
        session["user_type"] = user["user_type"]
        session["role"]      = int(user["role"])
        return jsonify({"status": "success", "user_type": user["user_type"]})
    return jsonify({"status": "fail"}), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/session_info")
def session_info():
    return jsonify({
        "user_id":   session.get("user_id"),
        "user_type": session.get("user_type", "student"),
        "role":      session.get("role", 0),
    })


# ── Chat API ──────────────────────────────────────────────────────────────────
@app.route("/get", methods=["POST"])
def chatbot():
    try:
        data       = request.get_json()
        user_input = data.get("prompt")
        chat_id    = data.get("chat_id")

        if not user_input:
            return jsonify({"status": "error", "text": "⚠️ Bạn chưa nhập câu hỏi."})

        save_message(chat_id, "user", user_input)
        response = ask_rag(user_input)
        save_message(chat_id, "assistant", response)
        return jsonify({"status": "success", "text": response})
    except Exception as e:
        return jsonify({"status": "error", "text": str(e)})


# ── Chat management ───────────────────────────────────────────────────────────
@app.route("/list_chats")
def api_list_chats():
    if not logged_in():
        return jsonify([])
    owner_role = 1 if is_teacher() else 0
    return jsonify(get_all_chats(session["user_id"], owner_role))


@app.route("/create_chat", methods=["POST"])
def api_create_chat():
    if not logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    owner_role = 1 if is_teacher() else 0
    chat_id = create_chat(session["user_id"], owner_role)
    return jsonify({"chat_id": chat_id})


@app.route("/rename_chat", methods=["POST"])
def api_rename_chat():
    data = request.get_json()
    rename_chat(data["chat_id"], data["title"])
    return jsonify({"status": "ok"})


@app.route("/delete_chat", methods=["POST"])
def api_delete_chat():
    if not logged_in():
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    delete_chat(data["chat_id"])
    return jsonify({"status": "ok"})


@app.route("/get_chat_messages")
def api_get_messages():
    chat_id = request.args.get("chat_id")
    return jsonify(get_messages(chat_id))


# ── Import law ────────────────────────────────────────────────────────────────
@app.route("/import_law", methods=["POST"])
def import_law():
    if not logged_in() or not is_teacher():
        return jsonify({"status": "error", "message": "Unauthorized"}), 403

    so_ky_hieu     = request.form.get("so_ky_hieu", "").strip()
    loai_van_ban   = request.form.get("loai_van_ban", "").strip()
    nguon_thu_thap = request.form.get("nguon_thu_thap", "").strip()
    pdf_file       = request.files.get("pdf_file")

    if not pdf_file or not so_ky_hieu or not loai_van_ban or not nguon_thu_thap:
        return jsonify({
            "status": "error",
            "message": "Vui lòng tải lên file PDF và điền đầy đủ tất cả 3 trường."
        }), 400

    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({"status": "error", "message": "Chỉ chấp nhận file PDF."}), 400

    job_id   = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    pdf_file.save(pdf_path)

    teacher_id = session["user_id"]

    def bg():
        run_import(
            job_id         = job_id,
            pdf_path       = pdf_path,
            so_ky_hieu     = so_ky_hieu,
            loai_van_ban   = loai_van_ban,
            nguon_thu_thap = nguon_thu_thap,
            student_id     = teacher_id,
            db_conn_factory = get_conn
        )

    threading.Thread(target=bg, daemon=True).start()

    return jsonify({"status": "ok", "job_id": job_id,
                    "message": "Đã nhận file. Đang xử lý nền…"})


@app.route("/import_status/<job_id>")
def import_status(job_id):
    job = get_job(job_id)
    return jsonify(job if job else {"status": "unknown"})


# ── Voice routes ─────────────────────────────────────────────────────────────
@app.route("/voice/transcribe", methods=["POST"])
def voice_transcribe():
    if _VOICE_ERROR:
        return jsonify({"error": f"Voice not available: {_VOICE_ERROR}. Run: pip install -r requirements_voice.txt"}), 503
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401
    if "audio" not in request.files:
        return jsonify({"error": "no audio file"}), 400

    audio_file  = request.files["audio"]
    audio_bytes = audio_file.read()
    mime        = audio_file.content_type or "audio/webm"

    import time
    t_start = time.perf_counter()
    result  = transcribe_audio(audio_bytes, mime)
    result["latency_total_ms"] = round((time.perf_counter() - t_start) * 1000)
    return jsonify(result)


@app.route("/voice/speak", methods=["POST"])
def voice_speak():
    if _VOICE_ERROR:
        return jsonify({"error": f"Voice not available: {_VOICE_ERROR}"}), 503
    if "user_id" not in session:
        return jsonify({"error": "not logged in"}), 401

    data        = request.get_json(force=True)
    answer_text = data.get("text", "").strip()
    voice       = data.get("voice", None)
    use_rvc     = data.get("use_rvc", True)

    if not answer_text:
        return jsonify({"error": "no text provided"}), 400

    result   = speak_answer(answer_text, voice=voice, use_rvc=use_rvc)
    response = app.response_class(
        response=result["audio"],
        status=200,
        mimetype=result["mime"],
    )
    response.headers["X-Latency-TTS-ms"] = result["latency_tts_ms"]
    response.headers["X-Latency-RVC-ms"] = result["latency_rvc_ms"]
    response.headers["X-RVC-Used"]        = str(result["rvc_used"]).lower()
    return response


@app.route("/voice/status")
def voice_status():
    from voice import rvc, tts
    return jsonify({
        "whisper_model": os.getenv("WHISPER_MODEL", "small"),
        "tts_voice":     tts.DEFAULT_VOICE,
        "rvc_endpoint":  rvc.RVC_ENDPOINT or None,
        "rvc_available": rvc.is_available(),
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)