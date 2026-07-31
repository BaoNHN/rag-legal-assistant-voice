from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks, Query, Header
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import os
import re
import uuid
import io

from engine.rag_engine import (
    ask_rag,
    list_indexed_sources, delete_source,
    list_dataset_sources, delete_dataset_source,
    list_scenario_sources, delete_scenario_source,
)
from database.database import (
    init_db, get_conn,
    login_user,
    create_chat, get_all_chats,
    save_message, get_messages,
    rename_chat, delete_chat,
    get_all_users, set_user_status, delete_user,
    change_user_password,
    create_voice_notification, list_voice_notifications,
    count_unread_voice_notifications, mark_voice_notifications_read,
)
from engine.import_law_engine import run_import, get_job
from engine.import_scenario_engine import run_import_scenario, get_scenario_job
from engine.import_dataset_engine import run_import_dataset, get_dataset_job
from engine.evaluate_engine import run_evaluation, get_eval_job, list_available_datasets, get_latest_eval_result
from engine.import_account_engine import run_import_accounts, build_template_bytes
from voice import station_client
from voice.station_client import MIN_TRAIN_SAMPLES, MAX_CLONED_VOICES_PER_USER, VoiceStationError

PASSWORD_RE = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$')

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="secret_key", max_age=7200)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads_tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

init_db()

# Self-register so clone-voice-station can push voice_profile_deleted/disabled
# events to us instead of us having to poll for them (see POST /voice/webhook
# below and voice/station_client.py's poll_undelivered_notifications as the
# fallback for whenever this push failed or was never set up). Best-effort —
# the station being down at startup must not block this app from starting.
SELF_BASE_URL = os.getenv("SELF_BASE_URL", "http://127.0.0.1:8000")
try:
    station_client.register_webhook(f"{SELF_BASE_URL}/voice/webhook")
except Exception:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def logged_in(request: Request) -> bool:
    return "user_id" in request.session

def is_teacher(request: Request) -> bool:
    # Admins have all Teacher-role functionality too.
    return request.session.get("role", 0) in (1, 2)

def is_admin(request: Request) -> bool:
    return request.session.get("role", 0) == 2


# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not logged_in(request):
        return templates.TemplateResponse(request, "login.html")
    return templates.TemplateResponse(request, "index.html", {
        "is_teacher": is_teacher(request),
        "is_admin":   is_admin(request),
    })


@app.get("/voice", response_class=HTMLResponse)
async def voice_page(request: Request):
    if not logged_in(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "voice_profile.html", {
        "min_samples": MIN_TRAIN_SAMPLES,
        "max_profiles": MAX_CLONED_VOICES_PER_USER,
    })


@app.get("/admin/voice_models", response_class=HTMLResponse)
async def admin_voice_models_page(request: Request):
    if not logged_in(request) or not is_admin(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "admin_voice_models.html", {"min_samples": MIN_TRAIN_SAMPLES})


@app.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    if not logged_in(request) or not is_teacher(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "import_law.html")


@app.get("/manage_accounts", response_class=HTMLResponse)
async def manage_accounts_page(request: Request):
    if not logged_in(request) or not is_admin(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "manage_accounts.html")


@app.get("/manage_law", response_class=HTMLResponse)
async def manage_law_page(request: Request):
    if not logged_in(request) or not is_admin(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "manage_law.html")


@app.get("/import_account", response_class=HTMLResponse)
async def import_account_page(request: Request):
    if not logged_in(request) or not is_admin(request):
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse(request, "import_account.html")


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/login")
async def login(request: Request):
    data     = await request.json()
    username = data.get("student_name") or data.get("username", "")
    password = data.get("password", "")

    user = login_user(username, password)
    if user and user.get("disabled"):
        return JSONResponse(
            {"status": "fail", "message": "Tài khoản đã bị vô hiệu hóa."},
            status_code=403
        )
    if user:
        request.session["user_id"]   = user["user_id"]
        request.session["user_type"] = user["user_type"]
        request.session["role"]      = int(user["role"])
        return {"status": "success", "user_type": user["user_type"]}
    return JSONResponse({"status": "fail"}, status_code=401)


@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


@app.get("/session_info")
async def session_info(request: Request):
    user_id = request.session.get("user_id")
    return {
        "user_id":       user_id,
        "user_type":     request.session.get("user_type", "student"),
        "role":          request.session.get("role", 0),
        "voice_consent": station_client.has_voice_consent(str(user_id)) if user_id else False,
    }


# ── Chat API ──────────────────────────────────────────────────────────────────
@app.post("/get")
async def chatbot(request: Request):
    try:
        data       = await request.json()
        user_input = data.get("prompt")
        chat_id    = data.get("chat_id")

        if not user_input:
            return {"status": "error", "text": "⚠️ Bạn chưa nhập câu hỏi."}

        save_message(chat_id, "user", user_input)
        response = ask_rag(user_input)
        save_message(chat_id, "assistant", response)
        return {"status": "success", "text": response}
    except Exception as e:
        return {"status": "error", "text": str(e)}


# ── Chat management ───────────────────────────────────────────────────────────
@app.get("/list_chats")
async def api_list_chats(request: Request):
    if not logged_in(request):
        return []
    owner_role = 1 if is_teacher(request) else 0
    return get_all_chats(request.session["user_id"], owner_role)


@app.post("/create_chat")
async def api_create_chat(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    owner_role = 1 if is_teacher(request) else 0
    chat_id = create_chat(request.session["user_id"], owner_role)
    return {"chat_id": chat_id}


@app.post("/rename_chat")
async def api_rename_chat(request: Request):
    data = await request.json()
    rename_chat(data["chat_id"], data["title"])
    return {"status": "ok"}


@app.post("/delete_chat")
async def api_delete_chat(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    data = await request.json()
    chat_id = data.get("chat_id")
    if not chat_id:
        return JSONResponse({"error": "Missing chat_id"}, status_code=400)
    delete_chat(chat_id)
    return {"status": "ok"}


@app.get("/get_chat_messages")
async def api_get_messages(chat_id: str = Query(...)):
    return get_messages(chat_id)


# ── Import law ────────────────────────────────────────────────────────────────
@app.post("/import_law")
async def import_law(
    request: Request,
    background_tasks: BackgroundTasks,
    so_ky_hieu: str = Form(""),
    loai_van_ban: str = Form(""),
    nguon_thu_thap: str = Form(""),
    pdf_file: UploadFile = File(None),
):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    so_ky_hieu     = so_ky_hieu.strip()
    loai_van_ban   = loai_van_ban.strip()
    nguon_thu_thap = nguon_thu_thap.strip()

    if not pdf_file or not so_ky_hieu or not loai_van_ban or not nguon_thu_thap:
        return JSONResponse({
            "status": "error",
            "message": "Vui lòng tải lên file PDF hoặc DOCX và điền đầy đủ tất cả 3 trường."
        }, status_code=400)

    filename_lower = (pdf_file.filename or "").lower()
    if not (filename_lower.endswith(".pdf") or filename_lower.endswith(".docx")):
        return JSONResponse({"status": "error", "message": "Chỉ chấp nhận file PDF hoặc DOCX."}, status_code=400)

    ext      = ".docx" if filename_lower.endswith(".docx") else ".pdf"
    job_id   = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    content = await pdf_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    teacher_id = request.session["user_id"]

    background_tasks.add_task(
        run_import,
        job_id=job_id,
        file_path=file_path,
        so_ky_hieu=so_ky_hieu,
        loai_van_ban=loai_van_ban,
        nguon_thu_thap=nguon_thu_thap,
        student_id=teacher_id,
        db_conn_factory=get_conn,
    )

    return {"status": "ok", "job_id": job_id, "message": "Đã nhận file. Đang xử lý nền…"}


@app.get("/import_status/{job_id}")
async def import_status(job_id: str):
    job = get_job(job_id)
    return job if job else {"status": "unknown"}


@app.get("/list_law_sources")
async def list_law_sources_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    return list_indexed_sources()


@app.post("/delete_law_source")
async def delete_law_source_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    data       = await request.json()
    so_ky_hieu = (data.get("so_ky_hieu") or "").strip()
    if not so_ky_hieu:
        return JSONResponse({"status": "error", "message": "Thiếu so_ky_hieu"}, status_code=400)

    deleted = delete_source(so_ky_hieu)
    if deleted == 0:
        return JSONResponse(
            {"status": "error", "message": f"Không tìm thấy văn bản '{so_ky_hieu}' trong ChromaDB."},
            status_code=404,
        )
    return {"status": "ok", "deleted": deleted}


@app.get("/list_dataset_sources")
async def list_dataset_sources_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    return list_dataset_sources()


@app.post("/delete_dataset_source")
async def delete_dataset_source_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    data        = await request.json()
    source_name = (data.get("name") or "").strip()
    if not source_name:
        return JSONResponse({"status": "error", "message": "Thiếu name"}, status_code=400)

    deleted = delete_dataset_source(source_name)
    if deleted == 0:
        return JSONResponse(
            {"status": "error", "message": f"Không tìm thấy dataset '{source_name}' trong ChromaDB."},
            status_code=404,
        )
    return {"status": "ok", "deleted": deleted}


@app.get("/list_scenario_sources")
async def list_scenario_sources_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    return list_scenario_sources()


@app.post("/delete_scenario_source")
async def delete_scenario_source_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    data        = await request.json()
    source_name = (data.get("name") or "").strip()
    if not source_name:
        return JSONResponse({"status": "error", "message": "Thiếu name"}, status_code=400)

    deleted = delete_scenario_source(source_name)
    if deleted == 0:
        return JSONResponse(
            {"status": "error", "message": f"Không tìm thấy tình huống '{source_name}' trong ChromaDB."},
            status_code=404,
        )
    return {"status": "ok", "deleted": deleted}


# ── Import scenario (DOCX case-study set) ───────────────────────────────────────
@app.post("/import_scenario")
async def import_scenario_route(
    request: Request,
    background_tasks: BackgroundTasks,
    docx_file: UploadFile = File(None),
):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    if not docx_file or not (docx_file.filename or "").lower().endswith(".docx"):
        return JSONResponse({"status": "error", "message": "Chỉ chấp nhận file .docx"}, status_code=400)

    job_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.docx")
    content   = await docx_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    teacher_id = request.session["user_id"]

    background_tasks.add_task(
        run_import_scenario,
        job_id=job_id,
        file_path=file_path,
        student_id=teacher_id,
        original_filename=docx_file.filename,
    )
    return {"status": "ok", "job_id": job_id, "message": "Đã nhận file. Đang xử lý nền…"}


@app.get("/import_scenario_status/{job_id}")
async def import_scenario_status(job_id: str):
    job = get_scenario_job(job_id)
    return job if job else {"status": "unknown"}


@app.get("/download_scenario_example")
async def download_scenario_example_route(request: Request):
    if not logged_in(request) or not is_teacher(request):
        return RedirectResponse("/", status_code=302)

    file_path = os.path.join(BASE_DIR, "Dataset", "example_scenario.docx")
    if not os.path.exists(file_path):
        return JSONResponse({"status": "error", "message": "Không tìm thấy file mẫu"}, status_code=404)

    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=example_scenario.docx"},
    )


# ── Import dataset (Excel) ─────────────────────────────────────────────────────
@app.post("/import_dataset")
async def import_dataset_route(
    request: Request,
    background_tasks: BackgroundTasks,
    dataset_file: UploadFile = File(None),
):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    if not dataset_file:
        return JSONResponse(
            {"status": "error", "message": "Vui lòng tải lên file dataset (.xlsx)"},
            status_code=400,
        )

    fname = (dataset_file.filename or "").lower()
    if not fname.endswith(".xlsx"):
        return JSONResponse(
            {"status": "error", "message": "Chỉ chấp nhận file .xlsx"},
            status_code=400,
        )

    job_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.xlsx")
    content   = await dataset_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    background_tasks.add_task(
        run_import_dataset,
        job_id=job_id,
        file_path=file_path,
        original_filename=dataset_file.filename,
    )
    return {"status": "ok", "job_id": job_id, "message": "Đang xử lý dataset…"}


@app.get("/import_dataset_status/{job_id}")
async def import_dataset_status(job_id: str):
    job = get_dataset_job(job_id)
    return job if job else {"status": "unknown"}


# ── Evaluate RAG ──────────────────────────────────────────────────────────────
@app.get("/list_datasets")
async def list_datasets_route(request: Request):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    return list_available_datasets()


@app.get("/download_dataset_example")
async def download_dataset_example_route(request: Request):
    if not logged_in(request) or not is_teacher(request):
        return RedirectResponse("/", status_code=302)

    file_path = os.path.join(BASE_DIR, "Dataset", "example_sheet.xlsx")
    if not os.path.exists(file_path):
        return JSONResponse({"status": "error", "message": "Không tìm thấy file mẫu"}, status_code=404)

    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=example_sheet.xlsx"},
    )


@app.post("/evaluate")
async def evaluate_route(
    request: Request,
    background_tasks: BackgroundTasks,
):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    data         = await request.json()
    mode         = data.get("mode", "auto")
    split        = data.get("split", "demo")
    dataset_file = data.get("dataset_file") or None

    if mode not in ("auto", "llm"):
        mode = "auto"
    if split not in ("demo", "all", "test"):
        split = "demo"

    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_evaluation, job_id=job_id, mode=mode, split=split, dataset_file=dataset_file)
    return {"status": "ok", "job_id": job_id}


@app.get("/evaluate_status/{job_id}")
async def evaluate_status_route(job_id: str):
    job = get_eval_job(job_id)
    return job if job else {"status": "unknown"}


@app.get("/latest_eval_result")
async def latest_eval_result_route(request: Request):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    result = get_latest_eval_result()
    return result if result else {"status": "empty"}


@app.get("/download_eval_result/{filename}")
async def download_eval_result_route(request: Request, filename: str):
    if not logged_in(request) or not is_teacher(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    safe_name = os.path.basename(filename)
    if not re.match(r'^eval_results_.*\.xlsx$', safe_name):
        return JSONResponse({"status": "error", "message": "Tên file không hợp lệ"}, status_code=400)

    file_path = os.path.join(BASE_DIR, safe_name)
    if not os.path.exists(file_path):
        return JSONResponse({"status": "error", "message": "Không tìm thấy file kết quả"}, status_code=404)

    return StreamingResponse(
        open(file_path, "rb"),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={safe_name}"},
    )


# ── Account management (admin) ─────────────────────────────────────────────────
@app.get("/list_users")
async def list_users_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    return get_all_users()


@app.post("/toggle_user_status")
async def toggle_user_status_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    data    = await request.json()
    user_id = data.get("user_id")
    status  = data.get("status")
    if user_id is None or status not in (0, 1):
        return JSONResponse({"status": "error", "message": "Thiếu tham số"}, status_code=400)
    if int(user_id) == request.session.get("user_id"):
        return JSONResponse({"status": "error", "message": "Không thể tự vô hiệu hóa tài khoản của chính mình."}, status_code=400)

    set_user_status(user_id, status)
    return {"status": "ok"}


@app.post("/delete_user")
async def delete_user_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    data    = await request.json()
    user_id = data.get("user_id")
    if user_id is None:
        return JSONResponse({"status": "error", "message": "Thiếu user_id"}, status_code=400)
    if int(user_id) == request.session.get("user_id"):
        return JSONResponse({"status": "error", "message": "Không thể tự xoá tài khoản của chính mình."}, status_code=400)

    delete_user(user_id)
    return {"status": "ok"}


@app.post("/import_account")
async def import_account_route(
    request: Request,
    account_file: UploadFile = File(None),
):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)

    if not account_file or not (account_file.filename or "").lower().endswith(".xlsx"):
        return JSONResponse({"status": "error", "message": "Chỉ chấp nhận file .xlsx"}, status_code=400)

    job_id    = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.xlsx")
    content   = await account_file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        report = run_import_accounts(file_path)
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass

    status_code = 200 if report.get("status") == "ok" else 400
    return JSONResponse(report, status_code=status_code)


@app.get("/download_account_template")
async def download_account_template(request: Request):
    if not logged_in(request) or not is_admin(request):
        return RedirectResponse("/", status_code=302)

    content = build_template_bytes()
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=account_import_template.xlsx"},
    )


# ── Change password (public, from login page) ──────────────────────────────────
@app.post("/change_password")
async def change_password_route(request: Request):
    data             = await request.json()
    username         = (data.get("username") or "").strip()
    old_password     = data.get("old_password") or ""
    new_password     = data.get("new_password") or ""
    confirm_password = data.get("confirm_password") or ""

    if not username or not old_password or not new_password or not confirm_password:
        return JSONResponse({"status": "error", "message": "Vui lòng điền đầy đủ tất cả các trường."}, status_code=400)
    if new_password != confirm_password:
        return JSONResponse({"status": "error", "message": "Mật khẩu mới và xác nhận mật khẩu không khớp."}, status_code=400)
    if not PASSWORD_RE.match(new_password):
        return JSONResponse({
            "status": "error",
            "message": "Mật khẩu mới cần tối thiểu 8 ký tự, gồm chữ hoa, chữ thường, số và ký tự đặc biệt."
        }, status_code=400)

    ok, reason = change_user_password(username, old_password, new_password)
    if not ok:
        return JSONResponse({"status": "error", "message": reason}, status_code=400)
    return {"status": "ok", "message": "Đổi mật khẩu thành công."}


# ── Voice: scripts + profiles (any logged-in user) ─────────────────────────────
# All of these delegate to clone-voice-station (voice/station_client.py) — this
# app only checks who's logged in / who's admin, ownership of a profile is
# enforced station-side by (client_id, external_user_id).
def _voice_error(e: VoiceStationError) -> JSONResponse:
    return JSONResponse({"status": "error", "message": e.message}, status_code=e.status_code)


@app.get("/voice/scripts")
async def voice_scripts_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return station_client.get_scripts()


@app.get("/voice/profiles")
async def voice_profiles_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return station_client.list_voice_profiles(str(request.session["user_id"]))


@app.post("/voice/consent")
async def voice_consent_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    station_client.record_voice_consent(str(request.session["user_id"]))
    return {"status": "ok"}


@app.post("/voice/profiles")
async def create_voice_profile_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    external_user_id = str(request.session["user_id"])

    data = await request.json()
    name = (data.get("name") or "").strip()
    if not name:
        return JSONResponse({"status": "error", "message": "Vui lòng đặt tên cho giọng nói."}, status_code=400)

    try:
        profile_id = station_client.create_voice_profile(external_user_id, name)
    except VoiceStationError as e:
        return _voice_error(e)
    return {"status": "ok", "profile_id": profile_id}


@app.put("/voice/profiles/{profile_id}")
async def update_voice_profile_route(profile_id: int, request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    external_user_id = str(request.session["user_id"])
    data             = await request.json()
    try:
        station_client.update_voice_profile(
            profile_id, external_user_id,
            name=data.get("name"), is_default=data.get("is_default"),
        )
    except VoiceStationError as e:
        return _voice_error(e)
    return {"status": "ok"}


@app.delete("/voice/profiles/{profile_id}")
async def delete_voice_profile_route(profile_id: int, request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        result = station_client.delete_voice_profile(profile_id, str(request.session["user_id"]))
    except VoiceStationError as e:
        return _voice_error(e)
    return result


@app.post("/voice/profiles/{profile_id}/samples")
async def upload_voice_sample_route(
    profile_id: int,
    request: Request,
    script_id: str = Form(...),
    audio: UploadFile = File(...),
):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    content = await audio.read()
    try:
        result = station_client.upload_voice_sample(
            profile_id, str(request.session["user_id"]), script_id,
            audio.filename or "sample.wav", content,
        )
    except VoiceStationError as e:
        return _voice_error(e)
    return result


@app.get("/voice/profiles/{profile_id}/samples")
async def list_voice_samples_route(profile_id: int, request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        return station_client.list_voice_samples(profile_id, str(request.session["user_id"]))
    except VoiceStationError as e:
        return _voice_error(e)


@app.delete("/voice/profiles/{profile_id}/samples/{sample_id}")
async def delete_voice_sample_route(profile_id: int, sample_id: int, request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        station_client.delete_voice_sample(profile_id, sample_id, str(request.session["user_id"]))
    except VoiceStationError as e:
        return _voice_error(e)
    return {"status": "ok"}


@app.post("/voice/profiles/{profile_id}/train")
async def train_voice_profile_route(profile_id: int, request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        result = station_client.train_voice_profile(profile_id, str(request.session["user_id"]))
    except VoiceStationError as e:
        return _voice_error(e)
    return result


@app.get("/voice/profiles/{profile_id}/status")
async def voice_profile_status_route(profile_id: int, request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    try:
        return station_client.get_voice_profile_status(profile_id, str(request.session["user_id"]))
    except VoiceStationError as e:
        return _voice_error(e)


@app.post("/voice/speak")
async def voice_speak_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    data       = await request.json()
    text       = (data.get("text") or "").strip()
    profile_id = data.get("profile_id")
    try:
        profile_id = int(profile_id) if profile_id not in (None, "") else None
    except (TypeError, ValueError):
        profile_id = None

    if not text:
        return JSONResponse({"error": "Không có nội dung để đọc."}, status_code=400)

    try:
        result = station_client.speak(text, str(request.session["user_id"]), profile_id)
    except VoiceStationError as e:
        return JSONResponse({"error": e.message}, status_code=e.status_code)
    return Response(content=result["audio"], media_type=result["mime"])


# ── Voice: admin management ─────────────────────────────────────────────────────
@app.get("/list_voice_models")
async def list_voice_models_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    try:
        profiles = station_client.list_all_voice_profiles()
    except VoiceStationError as e:
        return _voice_error(e)

    # clone-voice-station only knows an opaque external_user_id — resolve it back to
    # a real username here, since only this app has the actual users table.
    names_by_id = {str(u["user_id"]): u["user_name"] for u in get_all_users()}
    for p in profiles:
        p["owner_name"] = names_by_id.get(p.get("external_user_id"), "(đã xoá)")
    return profiles


@app.post("/admin/voice_models/{profile_id}/retrain")
async def admin_retrain_voice_model_route(profile_id: int, request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    try:
        result = station_client.admin_retrain_voice_model(profile_id)
    except VoiceStationError as e:
        return _voice_error(e)
    return result


@app.post("/admin/voice_models/{profile_id}/disable")
async def admin_disable_voice_model_route(profile_id: int, request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    try:
        result = station_client.admin_disable_voice_model(profile_id)
    except VoiceStationError as e:
        return _voice_error(e)
    return result


@app.delete("/admin/voice_models/{profile_id}")
async def admin_delete_voice_model_route(profile_id: int, request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    try:
        result = station_client.admin_delete_voice_model(profile_id)
    except VoiceStationError as e:
        return _voice_error(e)
    return result


@app.get("/admin/rvc_endpoint")
async def admin_get_rvc_endpoint_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    try:
        return station_client.get_rvc_endpoint()
    except VoiceStationError as e:
        return _voice_error(e)


@app.post("/admin/rvc_endpoint")
async def admin_set_rvc_endpoint_route(request: Request):
    if not logged_in(request) or not is_admin(request):
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=403)
    data     = await request.json()
    endpoint = (data.get("endpoint") or "").strip()
    try:
        return station_client.set_rvc_endpoint(endpoint)
    except VoiceStationError as e:
        return _voice_error(e)


# ── Voice: notifications (manager-triggered delete/disable → this user) ────────
@app.post("/voice/webhook")
async def voice_webhook_route(request: Request, x_api_key: str = Header(None)):
    """Called by clone-voice-station itself (not a logged-in user) whenever its
    manager deletes/disables one of our users' cloned voices. Authenticated by
    the same API key we use to call it — only the station and this app know it."""
    if not x_api_key or x_api_key != station_client.get_own_api_key():
        return JSONResponse({"status": "error", "message": "Unauthorized"}, status_code=401)

    data             = await request.json()
    external_user_id = data.get("external_user_id")
    message          = (data.get("message") or "").strip()
    try:
        user_id = int(external_user_id)
    except (TypeError, ValueError):
        return JSONResponse({"status": "error", "message": "Invalid external_user_id"}, status_code=400)
    if not message:
        return JSONResponse({"status": "error", "message": "Missing message"}, status_code=400)

    create_voice_notification(user_id, message)
    return {"status": "ok"}


@app.get("/voice/notifications")
async def voice_notifications_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    user_id = request.session["user_id"]

    # Best-effort fallback poll — picks up anything the webhook push missed
    # (e.g. this app was offline when the manager deleted/disabled a voice).
    try:
        pending = station_client.poll_undelivered_notifications(str(user_id))
        for n in pending:
            create_voice_notification(user_id, n["message"])
            station_client.ack_notification(n["id"])
    except Exception:
        pass

    return {
        "notifications": list_voice_notifications(user_id),
        "unread_count":  count_unread_voice_notifications(user_id),
    }


@app.post("/voice/notifications/read")
async def voice_notifications_read_route(request: Request):
    if not logged_in(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    mark_voice_notifications_read(request.session["user_id"])
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
