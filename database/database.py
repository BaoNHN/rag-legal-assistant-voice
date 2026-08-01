# database.py

import hashlib
import secrets
import sqlite3
import time
import os

PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    dk   = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = (stored or "").split("$")
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS)
    return secrets.compare_digest(dk.hex(), hash_hex)


def _is_hashed(stored: str) -> bool:
    """PBKDF2 hashes from _hash_password are always '<32 hex><\\$><64 hex>' --
    a real plaintext password matching that exact shape is not realistic."""
    if not stored or "$" not in stored:
        return False
    salt_hex, _, hash_hex = stored.partition("$")
    hexdigits = set("0123456789abcdef")
    return len(salt_hex) == 32 and len(hash_hex) == 64 and set(salt_hex + hash_hex) <= hexdigits

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # database/ → root
DB_NAME  = os.path.join(BASE_DIR, "chat.db")

# keyword.status — one column, four independent kinds (see init_db()'s
# `keyword` table comment): a "scoring" keyword (source_keyword primary/
# secondary tagging, _score_doc() boost — article-scoped only, see
# set_source_keywords), an "out-of-scope" keyword (engine.rag_engine.
# _is_out_of_scope()'s blocklist, formerly the hardcoded OUT_OF_SCOPE_KEYWORDS
# Python constant), or a "priority" keyword (kind="priority" — boosts one
# specific source directly, ADDITIVELY per matched keyword, when its own
# phrase matches the question; see _score_doc). Even values = active, odd =
# disabled; each kind is its own (active, disabled) pair rather than one
# shared toggle, same pattern as the original two.
#
# A "penalty" kind (kind="penalty", unconditional per-source deprioritization)
# existed 2026-07-29 through 2026-07-29 and was removed the same day: it
# needed to be BOTH applied (59/2020/QH14's Điều 26 competing against
# 168/2025/NĐ-CP's Điều 76/77 on ELU177/178) AND not applied (Điều 26 is the
# correct answer on ELS066/ELU170/ELU169 — same article, different
# questions) — no single per-source value satisfies both, since it can't
# discriminate between sibling articles of the same document. Replaced
# entirely by "priority" (boosts the correct article directly, additively,
# without ever touching a competing source's own score) plus per-article
# tagging (source_type="law_article", see _score_doc) — live-verified
# (2026-07-29) to cover every case the penalty used to handle, with two
# cases actually improving once the penalty's side effects were removed.
# Status values 4-7 (the old penalty range) are intentionally left unused
# rather than reassigned, so any historical data referencing them fails
# loudly instead of silently changing meaning.
KEYWORD_STATUS_ACTIVE            = 0  # scoring keyword, selectable for tagging
KEYWORD_STATUS_DISABLED          = 1  # scoring keyword, hidden from picker
KEYWORD_STATUS_OOS_ACTIVE        = 2  # out-of-scope keyword, currently blocking
KEYWORD_STATUS_OOS_DISABLED      = 3  # out-of-scope keyword, currently not blocking
KEYWORD_STATUS_PRIORITY_ACTIVE   = 8  # priority keyword (+15, additive per match), selectable for priority tagging
KEYWORD_STATUS_PRIORITY_DISABLED = 9

# name -> active status, for create_keyword()'s `kind` param
_KEYWORD_KIND_TO_STATUS = {
    "scoring":  KEYWORD_STATUS_ACTIVE,
    "oos":      KEYWORD_STATUS_OOS_ACTIVE,
    "priority": KEYWORD_STATUS_PRIORITY_ACTIVE,
}
# Additive per matched keyword (not a flat one-time bonus) — see _score_doc.
# +15 sized so 3-4 stacked matching keywords close the ~45-60 point gap
# measured on the hardest cases (2026-07-29), letting an admin close a gap of
# any size by tagging more genuinely-matching phrases rather than needing one
# single large number (per user request). Single tier, no hard/soft split —
# stack more keywords instead of picking a severity (per user request).
PRIORITY_SCORE = 15


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

    # Rename the legacy "Import new law" status chat to "Thông báo" — the
    # reserved title used from here on (see NOTIFICATION_CHAT_TITLE below).
    # NOTIFICATION_CHAT_TITLE isn't defined yet at this point in the file, but
    # init_db() only ever runs after the whole module has finished loading.
    c.execute("UPDATE chats SET title=? WHERE title=?", (NOTIFICATION_CHAT_TITLE, "Import new law"))

    # Scenario import used to post to its own separate status chat
    # ("Nhập văn bản tình huống") instead of the shared "Thông báo" chat —
    # redundant, and unlike NOTIFICATION_CHAT_TITLE it wasn't excluded from
    # MAX_CHATS_PER_USER or locked read-only in the UI. Fold any such legacy
    # chat's messages into the user's "Thông báo" chat (creating one if they
    # don't have it yet) and drop the now-empty legacy chat.
    c.execute("SELECT id, student_id FROM chats WHERE title=? AND role=1", ("Nhập văn bản tình huống",))
    for legacy_id, uid in c.fetchall():
        c.execute(
            "SELECT id FROM chats WHERE student_id=? AND title=? AND role=1 ORDER BY created_at DESC LIMIT 1",
            (uid, NOTIFICATION_CHAT_TITLE)
        )
        row = c.fetchone()
        if row:
            target_id = row[0]
        else:
            target_id = f"import_{int(time.time()*1000)}_{uid}"
            c.execute(
                "INSERT INTO chats (id, student_id, title, created_at, role) VALUES (?,?,?,?,?)",
                (target_id, uid, NOTIFICATION_CHAT_TITLE, time.time(), 1)
            )
        c.execute("UPDATE messages SET chat_id=? WHERE chat_id=?", (target_id, legacy_id))
        c.execute("DELETE FROM chats WHERE id=?", (legacy_id,))
        _trim_notification_messages(c, target_id)

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

    # ── const (key/value store for cross-cutting config — e.g. the whitelist
    # of legitimate document sources currently indexed in chroma_db, used to
    # verify a citation isn't naming a source that was never actually imported)
    c.execute("""
        CREATE TABLE IF NOT EXISTS const (
            name    TEXT PRIMARY KEY,
            content TEXT
        )
    """)

    # ── keyword (admin-managed scoring keywords — replaces hardcoded topic
    # lists in engine.rag_engine._score_doc so newly-imported law sources can
    # get the same relevance boosts as curated content without a code change)
    # status: 0 = active (selectable when tagging a source), 1 = disabled
    # (kept, never hard-deleted — deleting could orphan source_keyword rows
    # pointing at chunks still indexed in chroma_db)
    c.execute("""
        CREATE TABLE IF NOT EXISTS keyword (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            name   TEXT UNIQUE NOT NULL,
            status INTEGER DEFAULT 0
        )
    """)

    # ── source_keyword (which keywords are tagged to which source, and
    # whether each tag is primary or secondary). Keyed by (source_type,
    # source_key) rather than per-chunk since that's how each import type is
    # already grouped for the admin UI:
    #   source_type='law'      source_key=so_ky_hieu     (list_indexed_sources)
    #   source_type='dataset'  source_key=source_file    (list_dataset_sources)
    #   source_type='scenario' source_key=nguon_thu_thap (list_scenario_sources)
    # A 2-part generic key (instead of one column per import type) means a
    # future 4th import type needs no schema change — avoids rewriting every
    # chunk's chroma metadata whenever an admin edits a source's keywords.
    #
    # Migration: source_keyword originally shipped keyed by so_ky_hieu only
    # (law sources exclusively, this feature's first iteration). Any table
    # from that iteration is renamed, its rows migrated in as source_type=
    # 'law', then dropped — before the generalized CREATE TABLE below.
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='source_keyword'")
    if c.fetchone():
        c.execute("PRAGMA table_info(source_keyword)")
        existing_cols = {row[1] for row in c.fetchall()}
        if "source_type" not in existing_cols:
            c.execute("ALTER TABLE source_keyword RENAME TO source_keyword_legacy")
            c.execute("""
                CREATE TABLE source_keyword (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_key  TEXT NOT NULL,
                    keyword_id  INTEGER NOT NULL,
                    kind        TEXT NOT NULL,
                    UNIQUE(source_type, source_key, keyword_id)
                )
            """)
            c.execute("""
                INSERT INTO source_keyword (source_type, source_key, keyword_id, kind)
                SELECT 'law', so_ky_hieu, keyword_id, kind FROM source_keyword_legacy
            """)
            c.execute("DROP TABLE source_keyword_legacy")

    c.execute("""
        CREATE TABLE IF NOT EXISTS source_keyword (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            source_key  TEXT NOT NULL,
            keyword_id  INTEGER NOT NULL,
            kind        TEXT NOT NULL,
            UNIQUE(source_type, source_key, keyword_id)
        )
    """)

    # Seed a starting keyword set drawn from
    # Danh_muc_van_ban_duoi_luat_Luat_Doanh_nghiep_Luat_Thuong_mai.docx so the
    # admin has something to tag sources with immediately — only runs once
    # (idempotent, same pattern as the default-account seed above).
    c.execute("SELECT COUNT(*) FROM keyword")
    if c.fetchone()[0] == 0:
        _seed_keywords = [
            "đăng ký doanh nghiệp", "hộ kinh doanh", "doanh nghiệp nhà nước",
            "sở hữu chéo", "chứng khoán", "trái phiếu doanh nghiệp",
            "đầu tư nước ngoài", "thuế và hóa đơn điện tử",
            "lao động và bảo hiểm xã hội", "phá sản", "cạnh tranh",
            "xúc tiến thương mại", "khuyến mại", "quảng cáo thương mại",
            "hội chợ triển lãm thương mại", "kinh doanh dịch vụ logistics",
            "nhượng quyền thương mại", "sở giao dịch hàng hóa",
            "giám định thương mại", "văn phòng đại diện thương nhân nước ngoài",
            "chi nhánh thương nhân nước ngoài", "thương mại điện tử",
            "xử phạt vi phạm hành chính thương mại",
            "chuyển đổi loại hình doanh nghiệp", "quản lý vốn nhà nước tại doanh nghiệp",
        ]
        c.executemany(
            "INSERT OR IGNORE INTO keyword (name, status) VALUES (?, 0)",
            [(k,) for k in _seed_keywords]
        )

    # Seed the out-of-scope blocklist (formerly the hardcoded
    # OUT_OF_SCOPE_KEYWORDS Python constant in engine/rag_engine.py) — moved
    # into the same admin-editable keyword table so admins can add/disable
    # blocking terms without a code change, same rationale as the scoring
    # keywords above. Runs once, independent of the scoring-keyword seed gate
    # (keyed on whether any status=2/3 row exists yet, not total row count).
    # "nhà đất" added here — missing from the original hardcoded list (found
    # 2026-07-28: "Thủ tục mua bán nhà đất..." slipped through the filter).
    c.execute("SELECT COUNT(*) FROM keyword WHERE status IN (2, 3)")
    if c.fetchone()[0] == 0:
        _seed_out_of_scope_keywords = [
            "ly hôn", "li hôn", "hôn nhân", "gia đình", "ly thân", "kết hôn",
            "hình sự", "tội phạm", "khởi tố", "bắt giữ", "truy tố", "tù giam",
            "đất đai", "nhà ở", "nhà đất", "bất động sản", "quyền sử dụng đất",
            "bảo hiểm xã hội", "bảo hiểm y tế", "tai nạn lao động",
            "thuế thu nhập cá nhân", "thuế giá trị gia tăng", "thuế tiêu thụ",
            "hải quan", "xuất nhập khẩu", "hành chính công",
        ]
        c.executemany(
            "INSERT OR IGNORE INTO keyword (name, status) VALUES (?, 2)",
            [(k,) for k in _seed_out_of_scope_keywords]
        )

    # ── dataset_file (test/evaluation dataset registry — added 2026-07-28)
    # Dataset Excel uploads no longer get embedded into ChromaDB (a real
    # data-leakage risk was confirmed: the same question/answer rows used by
    # Quick/Full Evaluation were retrievable as real-answer context). Instead
    # an uploaded file is only saved to Dataset/ on disk and tracked here —
    # this table is what the Manage Law "Dataset" tab lists/deletes from, not
    # ChromaDB metadata (see engine.import_dataset_engine.run_import_dataset).
    c.execute("""
        CREATE TABLE IF NOT EXISTS dataset_file (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT UNIQUE NOT NULL,
            importer    TEXT,
            uploaded_at REAL
        )
    """)

    conn.commit()
    conn.close()

    _migrate_plaintext_passwords()


def _migrate_plaintext_passwords():
    """One-time upgrade: passwords used to be stored/compared in plaintext.
    Since the plaintext is still sitting right there in the column, this hashes
    it in place -- no user needs to reset anything, login keeps working the
    same way. Safe to run on every startup: already-hashed rows are detected
    via _is_hashed() and left untouched."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    rows = c.execute("SELECT user_id, password FROM users").fetchall()
    migrated = 0
    for user_id, password in rows:
        if password and not _is_hashed(password):
            c.execute("UPDATE users SET password=? WHERE user_id=?", (_hash_password(password), user_id))
            migrated += 1
    if migrated:
        conn.commit()
        print(f"[auth] Migrated {migrated} plaintext password(s) to PBKDF2 hashes.")
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
# CHAT LIMITS / RESERVED NAMES
# =========================
# The teacher/admin "import law" status chat is reserved under this exact
# title — it's excluded from the 5-chat cap and users may not create/rename
# any chat to exactly this string (substrings like "Thông báo họp" are fine).
NOTIFICATION_CHAT_TITLE = "Thông báo"
MAX_CHATS_PER_USER = 5
NOTIFICATION_KEEP_LATEST = 5
# Per-chat cap on user turns (question/answer pairs) — applies equally to
# logged-in chats (counted in messages) and guest sessions (counted in the
# signed session cookie, never written to chat.db). See app.py's /get route.
MAX_MESSAGES_PER_CHAT = 15


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
        "SELECT user_id, role, status, password FROM users WHERE user_name=?",
        (username,)
    ).fetchone()
    conn.close()

    if not row or not _verify_password(password, row[3]):
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
        (chat_id, user_id, "Đoạn chat mới", time.time(), owner_role)
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


def rename_chat(chat_id, title) -> bool:
    """Returns False (no-op) if `title` is exactly the reserved
    NOTIFICATION_CHAT_TITLE — only an exact match is blocked, a title merely
    containing the word (e.g. "Thông báo họp lúc 9h") is fine."""
    if (title or "").strip() == NOTIFICATION_CHAT_TITLE:
        return False
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()
    return True


def get_chat_title(chat_id) -> str | None:
    conn = sqlite3.connect(DB_NAME)
    row  = conn.execute("SELECT title FROM chats WHERE id=?", (chat_id,)).fetchone()
    conn.close()
    return row[0] if row else None


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


def count_user_messages(chat_id) -> int:
    """Number of user turns already sent in this chat — used to enforce
    MAX_MESSAGES_PER_CHAT."""
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM messages WHERE chat_id=? AND role='user'", (chat_id,))
    count = c.fetchone()[0]
    conn.close()
    return count


def delete_chat(chat_id: str):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    c.execute("DELETE FROM chats    WHERE id=?",      (chat_id,))
    conn.commit()
    conn.close()


def _trim_notification_messages(c, chat_id: str, keep: int = NOTIFICATION_KEEP_LATEST):
    """Delete all but the `keep` most recent messages in a notification chat —
    these chats only ever accumulate (one import = one more message, forever),
    so without a cap they'd grow unbounded."""
    c.execute(
        "DELETE FROM messages WHERE chat_id=? AND id NOT IN ("
        "  SELECT id FROM messages WHERE chat_id=? ORDER BY timestamp DESC LIMIT ?"
        ")",
        (chat_id, chat_id, keep)
    )


def upsert_import_chat(user_id: int, message: str, title: str = NOTIFICATION_CHAT_TITLE):
    """Create the given teacher-chat title if missing, append message to it,
    then trim to the latest NOTIFICATION_KEEP_LATEST messages."""
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
    _trim_notification_messages(c, chat_id)
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
            (user_name, _hash_password(password), role)
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
    if not _verify_password(old_password, user["password"]):
        return False, "Mật khẩu cũ không đúng"

    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE users SET password=? WHERE user_name=?", (_hash_password(new_password), username))
    conn.commit()
    conn.close()
    return True, ""


# =========================
# KEYWORD (admin-managed scoring keywords — see engine.rag_engine._score_doc)
# =========================
# A keyword is meant to be a short matchable phrase, not a descriptive
# sentence — one admin-created entry ("văn bản gốc chưa hợp nhất (ưu tiên
# thấp hơn 67/VBHN-VPQH)", 57 chars) overflowed the Quản lý từ khóa table's
# row layout (2026-07-29, user report). 50 comfortably covers the longest
# legitimate existing keyword ("văn phòng đại diện thương nhân nước ngoài",
# 41 chars) while rejecting sentence-length descriptions like that one.
MAX_KEYWORD_NAME_LENGTH = 50


def create_keyword(name: str, kind: str = "scoring"):
    """Creates a new active keyword. kind: "scoring" (default, for
    article-scoped source tagging) | "oos" (out-of-scope blocklist) |
    "priority" (see KEYWORD_STATUS_* above). Returns its id, or None if the
    name already exists (one shared unique name namespace across every kind —
    a name can't be more than one kind at once) or exceeds
    MAX_KEYWORD_NAME_LENGTH."""
    name = (name or "").strip()
    if not name or len(name) > MAX_KEYWORD_NAME_LENGTH:
        return None
    status = _KEYWORD_KIND_TO_STATUS.get(kind, KEYWORD_STATUS_ACTIVE)
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    try:
        c.execute("INSERT INTO keyword (name, status) VALUES (?, ?)", (name, status))
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_all_keywords() -> list:
    """Every keyword of every kind/status — for the admin 'Từ khóa' tab
    (scoring + out-of-scope + priority, active + disabled; the UI derives
    "Loại" from status//2 (0=scoring,1=oos,4=priority — 2/3 reserved,
    formerly penalty, retired 2026-07-29) and "Trạng thái" from status being
    even vs odd)."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute("SELECT id, name, status FROM keyword ORDER BY name ASC").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "status": int(r[2] or 0)} for r in rows]


def get_active_keywords() -> list:
    """Active scoring keywords only (status=0) — for the article-scoped
    tag-picker <select> when tagging a source. Disabled, out-of-scope, AND
    priority keywords must not be selectable here — a priority keyword
    getting picked as a scoring tag by mistake was the exact confusion this
    status split was added to prevent (2026-07-29, user report)."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, name FROM keyword WHERE status=0 ORDER BY name ASC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1]} for r in rows]


def get_active_priority_keywords() -> list:
    """Active priority keywords only (status=8) — for the dedicated "Từ khóa
    tăng ưu tiên" tag-picker <select>. A priority tag boosts one specific
    source directly (see _score_doc) — added 2026-07-29 after a source-wide
    penalty on 59/2020/QH14 turned out to conflict with itself (same
    article legitimately correct on some questions, wrong on others;
    boosting the actually-correct competing article instead sidesteps that
    entirely, since it never touches the
    other source's own score). Single tier (no hard/soft) — stack more
    matching keywords instead of picking a severity."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, name FROM keyword WHERE status=8 ORDER BY name ASC"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1]} for r in rows]


def get_active_out_of_scope_keywords() -> list:
    """Active out-of-scope blocklist phrases (status=2) — used by
    engine.rag_engine._is_out_of_scope() in place of the old hardcoded
    OUT_OF_SCOPE_KEYWORDS Python constant (still kept there as a fallback if
    this DB read ever fails)."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT name FROM keyword WHERE status=2 ORDER BY name ASC"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_or_create_keyword(name: str) -> int:
    """Looks up a keyword by exact name, creating it (active) if missing.
    Used by Dataset/Scenario imports to auto-tag from their own curated
    retrieval_keywords / "Từ khóa" fields instead of a manual picker — those
    phrases won't generally already exist in the admin-seeded keyword list.
    Silently truncated to MAX_KEYWORD_NAME_LENGTH (unlike create_keyword,
    which rejects — this path runs unattended during import, with no form to
    show a validation error on)."""
    name = (name or "").strip()[:MAX_KEYWORD_NAME_LENGTH].strip()
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    row  = c.execute("SELECT id FROM keyword WHERE name=?", (name,)).fetchone()
    if row:
        conn.close()
        return row[0]

    c.execute("INSERT INTO keyword (name, status) VALUES (?, 0)", (name,))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return new_id


def set_keyword_status(keyword_id: int, status: int):
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("UPDATE keyword SET status=? WHERE id=?", (status, keyword_id))
    conn.commit()
    conn.close()


def get_source_keywords(source_type: str, source_key: str) -> dict:
    """{"primary": [{id,name}, ...], "secondary": [...], "priority":
    [{id,name}, ...]} for one source (source_type: 'law'|'law_article'|
    'dataset'|'scenario', source_key: so_ky_hieu / "<so_ky_hieu>#<article>" /
    source_file / nguon_thu_thap respectively — see init_db()'s comment on
    source_keyword for the mapping). "priority" tags boost the source
    directly, additively, when the keyword's own phrase matches the question
    (single tier, no severity) — see engine.rag_engine._score_doc.
    Deliberately not filtered by keyword.status — a keyword already tagged to
    a source stays tagged (and keeps scoring) even after being disabled;
    disabling only removes it from the picker for *new* tags."""
    source_key = (source_key or "").strip()
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        """
        SELECT k.id, k.name, sk.kind, k.status
        FROM source_keyword sk
        JOIN keyword k ON k.id = sk.keyword_id
        WHERE sk.source_type = ? AND sk.source_key = ?
        ORDER BY k.name ASC
        """,
        (source_type, source_key)
    ).fetchall()
    conn.close()
    result = {"primary": [], "secondary": [], "priority": []}
    for kid, name, kind, kw_status in rows:
        if kind == "priority":
            result["priority"].append({"id": kid, "name": name})
        elif kind == "primary":
            result["primary"].append({"id": kid, "name": name})
        else:
            result["secondary"].append({"id": kid, "name": name})
    return result


def get_tagged_articles(so_ky_hieu: str) -> list:
    """Lists every Điều of one law/document (source_type='law_article',
    source_key='<so_ky_hieu>#<article>') that has at least one tag, so the
    admin UI can show "which articles already have something set" instead of
    requiring them to guess/type an article number one at a time to find out
    (2026-07-29 UX request — the per-article picker only reveals what's
    tagged *after* you already know which Điều to look up, which made
    duplicate/forgotten tags easy to miss). Returns
    [{"article": "77", "primary": [names], "secondary": [names],
    "priority": [names]}, ...] sorted numerically by article number where
    it parses as an int, non-numeric suffixes (e.g. "20a") last alphabetically."""
    so_ky_hieu = (so_ky_hieu or "").strip()
    if not so_ky_hieu:
        return []
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        """
        SELECT sk.source_key, k.name, sk.kind
        FROM source_keyword sk
        JOIN keyword k ON k.id = sk.keyword_id
        WHERE sk.source_type = 'law_article'
        """
    ).fetchall()
    conn.close()

    prefix = so_ky_hieu + "#"
    by_article = {}
    for source_key, name, kind in rows:
        if not source_key.startswith(prefix):
            continue
        article = source_key[len(prefix):]
        bucket = by_article.setdefault(article, {"primary": [], "secondary": [], "priority": []})
        key = "priority" if kind == "priority" else ("primary" if kind == "primary" else "secondary")
        bucket[key].append(name)

    def sort_key(article):
        return (0, int(article)) if article.isdigit() else (1, article)

    return [
        {"article": article, **by_article[article]}
        for article in sorted(by_article, key=sort_key)
    ]


def set_source_keywords(source_type: str, source_key: str, primary_ids: list,
                        secondary_ids: list, priority_ids: list = None):
    """Replaces all keyword tags for one source in one shot (delete-then-
    insert — simpler than diffing, and this table is small per source).
    priority_ids tags the source as directly boosted (see get_source_keywords
    docstring) — optional, defaults to none so existing callers that only
    manage primary/secondary are unaffected."""
    source_key = (source_key or "").strip()
    if not source_key:
        return

    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("DELETE FROM source_keyword WHERE source_type=? AND source_key=?", (source_type, source_key))

    seen = set()
    for kid, kind in (
        [(kid, "primary") for kid in (primary_ids or [])]
        + [(kid, "priority") for kid in (priority_ids or [])]
        + [(kid, "secondary") for kid in (secondary_ids or [])]
    ):
        if kid in seen:
            continue
        seen.add(kid)
        c.execute(
            "INSERT INTO source_keyword (source_type, source_key, keyword_id, kind) VALUES (?,?,?,?)",
            (source_type, source_key, kid, kind)
        )

    conn.commit()
    conn.close()


def get_source_keywords_map() -> dict:
    """{source_type: {source_key: {"primary": {name,...}, "secondary": {...},
    "priority": {...}}}} for every tagged source — one bulk load per
    ask_rag() call, used by engine.rag_engine._score_doc() to boost sources
    whose admin-tagged keywords match (priority is additive per matched
    keyword, unlike primary/secondary's one-time bonus). Small table, no
    caching needed."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        """
        SELECT sk.source_type, sk.source_key, k.name, sk.kind, k.status
        FROM source_keyword sk
        JOIN keyword k ON k.id = sk.keyword_id
        """
    ).fetchall()
    conn.close()

    result = {}
    for source_type, source_key, name, kind, kw_status in rows:
        by_key = result.setdefault(source_type, {})
        bucket = by_key.setdefault(source_key, {
            "primary": set(), "secondary": set(),
            "penalty_hard": set(), "penalty_soft": set(),
            "priority": set(),
        })
        if kind == "penalty":
            bucket["penalty_hard" if kw_status in (4, 5) else "penalty_soft"].add(name.lower())
        elif kind == "priority":
            bucket["priority"].add(name.lower())
        elif kind == "primary":
            bucket["primary"].add(name.lower())
        else:
            bucket["secondary"].add(name.lower())
    return result


# =========================
# DATASET FILE REGISTRY (test/evaluation datasets only — see init_db() comment;
# these files are never embedded into ChromaDB and never affect real answers)
# =========================
def register_dataset_file(filename: str, importer: str = "admin1"):
    """Records (or re-stamps) a dataset file saved to Dataset/ as a tracked
    evaluation dataset. Upsert — re-uploading the same filename just updates
    importer/uploaded_at rather than erroring on the UNIQUE constraint."""
    filename = (filename or "").strip()
    if not filename:
        return
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO dataset_file (filename, importer, uploaded_at) VALUES (?, ?, ?) "
        "ON CONFLICT(filename) DO UPDATE SET importer=excluded.importer, uploaded_at=excluded.uploaded_at",
        (filename, importer, time.time())
    )
    conn.commit()
    conn.close()


def get_all_dataset_files() -> list:
    """Every tracked dataset file — for the Manage Law 'Dataset' tab."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT filename, importer, uploaded_at FROM dataset_file ORDER BY filename ASC"
    ).fetchall()
    conn.close()
    return [{"filename": r[0], "importer": r[1] or "admin1", "uploaded_at": r[2]} for r in rows]


def delete_dataset_file(filename: str):
    """Removes a filename from the tracking table. Does not touch the file on
    disk — callers that also want the physical file gone should remove it
    from Dataset/ themselves."""
    filename = (filename or "").strip()
    if not filename:
        return
    conn = sqlite3.connect(DB_NAME)
    c    = conn.cursor()
    c.execute("DELETE FROM dataset_file WHERE filename=?", (filename,))
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