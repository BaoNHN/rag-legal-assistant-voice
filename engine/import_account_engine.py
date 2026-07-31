# import_account_engine.py
# Parses an Excel file (columns: "Account Name", "Role") and creates user
# accounts with role-based default passwords. Also builds the downloadable
# example template.

import base64
import io
import json
import os
import subprocess
import sys

from openpyxl import Workbook

from database.database import create_user

DEFAULT_PASSWORDS = {
    "student": "123456P@ss",
    "teacher": "Teacher@123",
}

_READER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_excel_row_reader.py")


def _read_excel_rows(file_path: str) -> list:
    """
    Reads the first sheet of an .xlsx file in an isolated subprocess.
    Parsing .xlsx with pandas inside this process (which already has
    torch/OpenCV loaded via the OCR/embedding engines) segfaults on this
    environment, so the read is delegated to a clean child process.
    """
    result = subprocess.run(
        [sys.executable, _READER_SCRIPT, file_path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Không thể đọc file Excel.")
    return json.loads(result.stdout)


def _normalize_role(raw_role: str):
    r = str(raw_role).strip().lower()
    if r in ("student", "sinh viên", "sinh vien", "sv", "0"):
        return "student", 0
    if r in ("teacher", "giảng viên", "giang vien", "gv", "1"):
        return "teacher", 1
    return None, None


def run_import_accounts(file_path: str) -> dict:
    """
    Reads the Excel file and creates one user per valid row.
    Returns a report dict: {created, skipped_duplicate, skipped_invalid, total, message}
    """
    try:
        rows = _read_excel_rows(file_path)
    except Exception as e:
        return {"status": "error", "message": f"Không thể đọc file Excel: {e}"}

    if not rows:
        return {"status": "error", "message": "File không có dữ liệu."}

    columns  = list(rows[0].keys())
    col_name = next((c for c in columns if c.strip().lower() == "account name"), None)
    col_role = next((c for c in columns if c.strip().lower() == "role"), None)

    if not col_name or not col_role:
        return {
            "status": "error",
            "message": 'File thiếu cột bắt buộc "Account Name" hoặc "Role".',
        }

    created, skipped_duplicate, skipped_invalid = 0, 0, 0
    error_rows = []

    for row in rows:
        name_cell    = row.get(col_name)
        role_raw     = row.get(col_role)
        account_name = "" if name_cell is None else str(name_cell).strip()
        role_display = "" if role_raw is None else str(role_raw).strip()

        if not account_name or role_raw is None:
            skipped_invalid += 1
            error_rows.append((account_name, role_display, "Thiếu tên tài khoản hoặc role"))
            continue

        role_label, role_num = _normalize_role(role_raw)
        if role_label is None:
            skipped_invalid += 1
            error_rows.append((account_name, role_display, "Role không hợp lệ"))
            continue

        password = DEFAULT_PASSWORDS[role_label]
        ok = create_user(account_name, password, role_num)
        if ok:
            created += 1
        else:
            skipped_duplicate += 1
            error_rows.append((account_name, role_display, "Tài khoản đã tồn tại (trùng tên)"))

    total = created + skipped_duplicate + skipped_invalid
    message = (
        f"✅ Hoàn tất!\n"
        f"Tạo mới: {created}\n"
        f"Bỏ qua (trùng tên): {skipped_duplicate}\n"
        f"Bỏ qua (không hợp lệ): {skipped_invalid}\n"
        f"Tổng số dòng: {total}"
    )

    error_report_b64 = None
    if error_rows:
        error_report_b64 = base64.b64encode(build_error_report_bytes(error_rows)).decode("ascii")

    return {
        "status": "ok",
        "created": created,
        "skipped_duplicate": skipped_duplicate,
        "skipped_invalid": skipped_invalid,
        "total": total,
        "message": message,
        "has_errors": bool(error_rows),
        "error_report_b64": error_report_b64,
    }


def build_template_bytes() -> bytes:
    """Builds an in-memory .xlsx template with columns Account Name / Role."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Accounts"
    ws.append(["Account Name", "Role"])
    ws.append(["student01", "Student"])
    ws.append(["teacher01", "Teacher"])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()


def build_error_report_bytes(error_rows: list) -> bytes:
    """Builds an in-memory .xlsx report of rows that failed to import.

    error_rows: list of (account_name, role, reason) tuples.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Loi"
    ws.append(["Account Name", "Role", "Nguyên nhân lỗi"])
    for account_name, role, reason in error_rows:
        ws.append([account_name, role, reason])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()
