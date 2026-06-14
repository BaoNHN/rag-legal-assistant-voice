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
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT UNIQUE,
            password  TEXT,
            role      INTEGER DEFAULT 0
        )
    """)

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

    conn.commit()
    conn.close()


# =========================
# LOGIN
# =========================
def login_user(username: str, password: str):
    """
    Returns dict: {user_id, user_name, user_type, role}
    role: 0=student, 1=teacher
    user_type: 'student' | 'teacher'
    Returns None if not found.
    """
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute(
        "SELECT user_id, role FROM users WHERE user_name=? AND password=?",
        (username, password)
    ).fetchone()
    conn.close()

    if not row:
        return None

    role      = int(row[1])
    user_type = "teacher" if role >= 1 else "student"
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


def upsert_import_chat(user_id: int, message: str):
    """Create 'Import new law' chat for teacher if missing, append message."""
    conn  = sqlite3.connect(DB_NAME)
    c     = conn.cursor()
    TITLE = "Import new law"

    c.execute(
        "SELECT id FROM chats WHERE student_id=? AND title=? AND role=1 ORDER BY created_at DESC LIMIT 1",
        (user_id, TITLE)
    )
    row = c.fetchone()
    if row:
        chat_id = row[0]
    else:
        chat_id = f"import_{int(time.time()*1000)}"
        c.execute(
            "INSERT INTO chats (id, student_id, title, created_at, role) VALUES (?,?,?,?,?)",
            (chat_id, user_id, TITLE, time.time(), 1)
        )
    c.execute(
        "INSERT INTO messages (chat_id, role, text, timestamp) VALUES (?,?,?,?)",
        (chat_id, "assistant", message, time.time())
    )
    conn.commit()
    conn.close()