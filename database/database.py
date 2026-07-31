# database.py

import sqlite3
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # database/ → root
DB_NAME  = os.path.join(BASE_DIR, "chat.db")


def get_conn():
    return sqlite3.connect(DB_NAME)


# =========================
# INIT DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()

    # ── users table (replaces students + teachers)
    # role: 0 = student, 1 = teacher, 2 = admin (extensible)
    # status: 0 = active, 1 = disabled (login restricted)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT UNIQUE,
            password  TEXT,
            role      INTEGER DEFAULT 0,
            status    INTEGER DEFAULT 0
        )
    """)
    # Migration for existing DBs missing status column
    try:
        c.execute("ALTER TABLE users ADD COLUMN status INTEGER DEFAULT 0")
    except Exception:
        pass

    # Migration for existing DBs missing voice_consent_at column
    # (timestamp the user accepted the voice-cloning disclaimer; NULL = not accepted)
    try:
        c.execute("ALTER TABLE users ADD COLUMN voice_consent_at REAL")
    except Exception:
        pass

    # ── chats
    # role: 0 = student chat, 1 = teacher chat
    # Filtering by (user_id, role) keeps teacher/student chats separate
    c.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id         TEXT PRIMARY KEY,
            student_id INTEGER,
            title      TEXT,
            created_at REAL,
            role       INTEGER DEFAULT 0
        )
    """)
    # Migration for existing DBs missing role column
    try:
        c.execute("ALTER TABLE chats ADD COLUMN role INTEGER DEFAULT 0")
    except Exception:
        pass

    # ── messages
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id   TEXT,
            role      TEXT,
            text      TEXT,
            timestamp REAL
        )
    """)

    # ── app_settings — generic key/value store.
    # Used for "rvc_endpoint" (the Colab tunnel URL), which changes every time
    # the Colab notebook is restarted, so it needs to be editable at runtime
    # from the admin UI rather than baked into an env var.
    c.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # ── voice_notifications — surfaced in the sidebar bell icon. Populated when
    # clone-voice-station's manager deletes/disables a user's cloned voice (see
    # POST /voice/webhook and voice/station_client.py's polling fallback).
    c.execute("""
        CREATE TABLE IF NOT EXISTS voice_notifications (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            message    TEXT,
            created_at REAL,
            read_at    REAL
        )
    """)

    # ── Migrate old students table → users (if exists)
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='students'")
    if c.fetchone():
        c.execute("SELECT student_id, student_name, password FROM students")
        old_students = c.fetchall()
        for sid, sname, spwd in old_students:
            try:
                c.execute(
                    "INSERT OR IGNORE INTO users (user_id, user_name, password, role) VALUES (?,?,?,0)",
                    (sid, sname, spwd)
                )
            except Exception:
                pass

    # ── Migrate old teachers table → users (if exists)
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='teachers'")
    if c.fetchone():
        c.execute("SELECT teacher_id, teacher_name, password FROM teachers")
        old_teachers = c.fetchall()
        for tid, tname, tpwd in old_teachers:
            try:
                c.execute(
                    "INSERT OR IGNORE INTO users (user_id, user_name, password, role) VALUES (?,?,?,1)",
                    (tid, tname, tpwd)
                )
            except Exception:
                pass

    # ── Seed default accounts if users table is empty
    c.execute("SELECT COUNT(*) FROM users")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO users (user_name, password, role) VALUES (?,?,?)",
            ("testStudent1", "123456P@ss", 0)
        )
        c.execute(
            "INSERT INTO users (user_name, password, role) VALUES (?,?,?)",
            ("teacher1", "Teacher@123", 1)
        )
        c.execute(
            "INSERT INTO users (user_name, password, role) VALUES (?,?,?)",
            ("admin1", "Admin@123", 2)
        )

    # Seed rvc_endpoint setting from env var on first run only — after that
    # it's owned by the admin UI (see set_setting / /admin/rvc_endpoint).
    c.execute("SELECT COUNT(*) FROM app_settings WHERE key='rvc_endpoint'")
    if c.fetchone()[0] == 0:
        env_endpoint = os.getenv("RVC_ENDPOINT", "").strip()
        c.execute(
            "INSERT INTO app_settings (key, value) VALUES ('rvc_endpoint', ?)",
            (env_endpoint,)
        )

    # ── const (key/value store for cross-cutting config — e.g. the whitelist
    # of legitimate document sources currently indexed in chroma_db, used to
    # verify a citation isn't naming a source that was never actually imported)
    c.execute("""
        CREATE TABLE IF NOT EXISTS const (
            name    TEXT PRIMARY KEY,
            content TEXT
        )
    """)

    conn.commit()
    conn.close()


# =========================
# CONST (key/value store)
# =========================
def _ensure_const_table(c):
    # engine.rag_engine calls set_const()/get_const() at module import time,
    # which can run before app.py's init_db() — don't depend on ordering.
    c.execute("""
        CREATE TABLE IF NOT EXISTS const (
            name    TEXT PRIMARY KEY,
            content TEXT
        )
    """)


def set_const(name: str, content: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    _ensure_const_table(c)
    c.execute(
        "INSERT INTO const (name, content) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET content=excluded.content",
        (name, content)
    )
    conn.commit()
    conn.close()


def get_const(name: str) -> str:
    conn = sqlite3.connect(DB_NAME)
    _ensure_const_table(conn.cursor())
    row  = conn.execute("SELECT content FROM const WHERE name=?", (name,)).fetchone()
    conn.commit()
    conn.close()
    return row[0] if row else ""


# =========================
# LOGIN
# =========================
ROLE_NAMES = {0: "Student", 1: "Teacher", 2: "Admin"}


def role_name(role: int) -> str:
    return ROLE_NAMES.get(int(role), "Student")


def login_user(username: str, password: str):
    """
    Returns dict: {user_id, user_name, user_type, role} on success.
    Returns {"disabled": True} if credentials match a disabled account.
    Returns None if credentials don't match any account.
    role: 0=student, 1=teacher, 2=admin
    user_type: 'student' | 'teacher' | 'admin'
    """
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT user_id, role, status FROM users WHERE user_name=? AND password=?",
        (username, password)
    ).fetchone()
    conn.close()

    if not row:
        return None

    role   = int(row[1])
    status = int(row[2] or 0)
    if status == 1:
        return {"disabled": True}

    user_type = "admin" if role == 2 else ("teacher" if role == 1 else "student")
    return {
        "user_id":   row[0],
        "user_name": username,
        "user_type": user_type,
        "role":      role,
    }


# =========================
# CHAT MANAGEMENT
# =========================
def create_chat(user_id, owner_role: int = 0):
    """
    owner_role: 0=student chat, 1=teacher chat.
    Stored in chats.role so get_all_chats filters correctly.
    """
    conn    = sqlite3.connect(DB_NAME)
    c       = conn.cursor()
    chat_id = f"chat_{int(time.time()*1000)}"
    c.execute(
        "INSERT INTO chats (id, student_id, title, created_at, role) VALUES (?,?,?,?,?)",
        (chat_id, user_id, "New Chat", time.time(), owner_role)
    )
    conn.commit()
    conn.close()
    return chat_id


def get_all_chats(user_id, owner_role: int = 0):
    """
    Filters by both user_id AND role — teachers and students
    never see each other's chats even if they share the same numeric id.
    """
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute(
            "SELECT id, title FROM chats WHERE student_id=? AND role=? ORDER BY created_at DESC",
            (user_id, owner_role)
        )
        return [{"id": r[0], "title": r[1]} for r in c.fetchall()]


def rename_chat(chat_id, title):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()


def save_message(chat_id, role, text):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO messages (chat_id, role, text, timestamp) VALUES (?,?,?,?)",
        (chat_id, role, str(text), time.time())
    )
    conn.commit()
    conn.close()


def get_messages(chat_id):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "SELECT role, text FROM messages WHERE chat_id=? ORDER BY timestamp ASC",
        (chat_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "text": r[1]} for r in rows]


def delete_chat(chat_id: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    c.execute("DELETE FROM chats    WHERE id=?",      (chat_id,))
    conn.commit()
    conn.close()


def upsert_import_chat(user_id: int, message: str, title: str = "Import new law"):
    """Create the given teacher-chat title if missing, append message to it."""
    conn  = sqlite3.connect(DB_NAME)
    c     = conn.cursor()

    c.execute(
        "SELECT id FROM chats WHERE student_id=? AND title=? AND role=1 ORDER BY created_at DESC LIMIT 1",
        (user_id, title)
    )
    row = c.fetchone()
    if row:
        chat_id = row[0]
    else:
        chat_id = f"import_{int(time.time()*1000)}"
        c.execute(
            "INSERT INTO chats (id, student_id, title, created_at, role) VALUES (?,?,?,?,?)",
            (chat_id, user_id, title, time.time(), 1)
        )
    c.execute(
        "INSERT INTO messages (chat_id, role, text, timestamp) VALUES (?,?,?,?)",
        (chat_id, "assistant", message, time.time())
    )
    conn.commit()
    conn.close()


# =========================
# ACCOUNT MANAGEMENT (admin)
# =========================
def get_all_users():
    """Returns list of {user_id, user_name, role, role_name, status} ordered by user_id."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("SELECT user_id, user_name, role, status FROM users ORDER BY user_id ASC")
    rows = c.fetchall()
    conn.close()
    return [
        {
            "user_id":   r[0],
            "user_name": r[1],
            "role":      int(r[2]),
            "role_name": role_name(r[2]),
            "status":    int(r[3] or 0),
        }
        for r in rows
    ]


def set_user_status(user_id: int, status: int):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE users SET status=? WHERE user_id=?", (status, user_id))
    conn.commit()
    conn.close()


def delete_user(user_id: int):
    """Deletes a user, all their chats (student + teacher role chats) and messages."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("SELECT id FROM chats WHERE student_id=?", (user_id,))
    chat_ids = [r[0] for r in c.fetchall()]
    for chat_id in chat_ids:
        c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    c.execute("DELETE FROM chats WHERE student_id=?", (user_id,))
    c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()


def create_user(user_name: str, password: str, role: int):
    """Creates a new user. Returns True if created, False if user_name already exists."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (user_name, password, role, status) VALUES (?,?,?,0)",
            (user_name, password, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_name(username: str):
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT user_id, user_name, password, role, status FROM users WHERE user_name=?",
        (username,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "user_id":   row[0],
        "user_name": row[1],
        "password":  row[2],
        "role":      int(row[3]),
        "status":    int(row[4] or 0),
    }


def change_user_password(username: str, old_password: str, new_password: str):
    """
    Verifies old_password against DB then updates to new_password.
    Returns (True, "") on success, or (False, reason) on failure.
    """
    user = get_user_by_name(username)
    if not user:
        return False, "Tài khoản không tồn tại"
    if user["password"] != old_password:
        return False, "Mật khẩu cũ không đúng"

    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE users SET password=? WHERE user_name=?", (new_password, username))
    conn.commit()
    conn.close()
    return True, ""


# =========================
# APP SETTINGS (key/value)
# =========================
def get_setting(key: str, default: str = "") -> str:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else default


def set_setting(key: str, value: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO app_settings (key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    conn.commit()
    conn.close()


# =========================
# VOICE NOTIFICATIONS (bell icon — see POST /voice/webhook, GET /voice/notifications)
# =========================
def create_voice_notification(user_id: int, message: str) -> int:
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO voice_notifications (user_id, message, created_at, read_at) VALUES (?,?,?,NULL)",
        (user_id, message, time.time())
    )
    notification_id = c.lastrowid
    conn.commit()
    conn.close()
    return notification_id


def list_voice_notifications(user_id: int, limit: int = 20):
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, message, created_at, read_at FROM voice_notifications "
        "WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "message": r[1], "created_at": r[2], "read": bool(r[3])} for r in rows]


def count_unread_voice_notifications(user_id: int) -> int:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT COUNT(*) FROM voice_notifications WHERE user_id=? AND read_at IS NULL", (user_id,)
    ).fetchone()
    conn.close()
    return row[0]


def mark_voice_notifications_read(user_id: int):
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "UPDATE voice_notifications SET read_at=? WHERE user_id=? AND read_at IS NULL",
        (time.time(), user_id)
    )
    conn.commit()
    conn.close()

