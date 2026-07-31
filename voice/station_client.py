"""
voice/station_client.py
HTTP client for clone-voice-station — the standalone service that now owns all
voice training/storage/management (see D:\\hoc\\project\\clone-voice-station).

This app authenticates itself to the station with a shared API key (read from
voice_station_key.txt) and identifies each end user as an opaque
external_user_id (str(session["user_id"])) — the station has no login of its
own and trusts this app to have already checked who's logged in / who's admin.

All functions degrade gracefully when the station is unreachable or the key
file is missing: reads return empty/None, writes raise VoiceStationError with
a message safe to show the user. Callers (app.py routes) turn that into a
clean error response instead of a 500.
"""

import os
import requests

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_PATH     = os.path.join(BASE_DIR, "voice_station_key.txt")
STATION_URL  = os.getenv("VOICE_STATION_URL", "http://127.0.0.1:8090").rstrip("/")

REQUEST_TIMEOUT       = int(os.getenv("VOICE_STATION_TIMEOUT", "15"))
SPEAK_TIMEOUT         = int(os.getenv("VOICE_STATION_SPEAK_TIMEOUT", "30"))
UPLOAD_TIMEOUT        = int(os.getenv("VOICE_STATION_UPLOAD_TIMEOUT", "30"))

# Display-only copies of clone-voice-station's own limits (it enforces these
# server-side regardless) — used to render voice_profile.html's hint text
# without an extra round-trip. Keep in sync with clone-voice-station/database/database.py.
MIN_TRAIN_SAMPLES           = 5
MAX_CLONED_VOICES_PER_USER  = 2


class VoiceStationError(Exception):
    """Raised on any failure talking to clone-voice-station — message is
    already user-safe Vietnamese text, ready to bubble up in a JSONResponse."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message     = message
        self.status_code = status_code


def _api_key() -> str:
    if not os.path.exists(KEY_PATH):
        return ""
    with open(KEY_PATH, "r") as f:
        return f.read().strip()


def get_own_api_key() -> str:
    """The key this app authenticates itself to the station with — also used
    to verify an inbound webhook call really came from the station, since it
    echoes this same key back in X-Api-Key (see app.py's POST /voice/webhook)."""
    return _api_key()


def _headers() -> dict:
    return {"X-Api-Key": _api_key()}


def _raise_for_response(resp):
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", f"HTTP {resp.status_code}")
        except Exception:
            detail = f"HTTP {resp.status_code}"
        raise VoiceStationError(detail, status_code=resp.status_code)


def _get(path: str, params: dict = None, timeout: int = REQUEST_TIMEOUT):
    try:
        resp = requests.get(f"{STATION_URL}{path}", headers=_headers(), params=params, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
    _raise_for_response(resp)
    return resp.json()


def _post(path: str, json: dict = None, timeout: int = REQUEST_TIMEOUT):
    try:
        resp = requests.post(f"{STATION_URL}{path}", headers=_headers(), json=json or {}, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
    _raise_for_response(resp)
    return resp.json()


def _put(path: str, json: dict = None, timeout: int = REQUEST_TIMEOUT):
    try:
        resp = requests.put(f"{STATION_URL}{path}", headers=_headers(), json=json or {}, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
    _raise_for_response(resp)
    return resp.json()


def _delete(path: str, params: dict = None, timeout: int = REQUEST_TIMEOUT):
    try:
        resp = requests.delete(f"{STATION_URL}{path}", headers=_headers(), params=params, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
    _raise_for_response(resp)
    return resp.json()


def is_available() -> bool:
    try:
        resp = requests.get(f"{STATION_URL}/api/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ── Scripts / consent / profiles ────────────────────────────────────────────────
def get_scripts() -> list:
    return _get("/api/scripts")


def has_voice_consent(external_user_id: str) -> bool:
    try:
        return bool(_get("/api/consent", params={"external_user_id": external_user_id}).get("consent"))
    except VoiceStationError:
        return False  # station unreachable — degrade to "no consent" rather than break login/session_info


def record_voice_consent(external_user_id: str):
    _post("/api/consent", {"external_user_id": external_user_id})


def list_voice_profiles(external_user_id: str) -> list:
    try:
        return _get("/api/profiles", params={"external_user_id": external_user_id})
    except VoiceStationError:
        return []


def create_voice_profile(external_user_id: str, name: str) -> int:
    return _post("/api/profiles", {"external_user_id": external_user_id, "name": name})["profile_id"]


def update_voice_profile(profile_id: int, external_user_id: str, name: str = None, is_default: bool = None):
    payload = {"external_user_id": external_user_id}
    if name is not None:
        payload["name"] = name
    if is_default is not None:
        payload["is_default"] = is_default
    return _put(f"/api/profiles/{profile_id}", payload)


def delete_voice_profile(profile_id: int, external_user_id: str) -> dict:
    return _delete(f"/api/profiles/{profile_id}", params={"external_user_id": external_user_id})


def get_voice_profile_status(profile_id: int, external_user_id: str) -> dict:
    return _get(f"/api/profiles/{profile_id}/status", params={"external_user_id": external_user_id})


# ── Samples ──────────────────────────────────────────────────────────────────
def upload_voice_sample(profile_id: int, external_user_id: str, script_id: str,
                         filename: str, content: bytes) -> dict:
    try:
        resp = requests.post(
            f"{STATION_URL}/api/profiles/{profile_id}/samples",
            headers=_headers(),
            data={"external_user_id": external_user_id, "script_id": script_id},
            files={"audio": (filename, content)},
            timeout=UPLOAD_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
    _raise_for_response(resp)
    return resp.json()


def list_voice_samples(profile_id: int, external_user_id: str) -> list:
    return _get(f"/api/profiles/{profile_id}/samples", params={"external_user_id": external_user_id})


def delete_voice_sample(profile_id: int, sample_id: int, external_user_id: str):
    return _delete(f"/api/profiles/{profile_id}/samples/{sample_id}", params={"external_user_id": external_user_id})


def train_voice_profile(profile_id: int, external_user_id: str) -> dict:
    return _post(f"/api/profiles/{profile_id}/train", {"external_user_id": external_user_id})


# ── Speak (TTS + optional RVC) ───────────────────────────────────────────────
def speak(text: str, external_user_id: str, profile_id: int = None) -> dict:
    """Returns {"audio": bytes, "mime": str}."""
    try:
        resp = requests.post(
            f"{STATION_URL}/api/speak",
            headers=_headers(),
            json={"text": text, "external_user_id": external_user_id, "profile_id": profile_id},
            timeout=SPEAK_TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        raise VoiceStationError(f"Không kết nối được tới voice station: {e}", status_code=503)
    _raise_for_response(resp)
    return {"audio": resp.content, "mime": resp.headers.get("content-type", "audio/mpeg")}


# ── Admin (client-wide, gated by is_admin() in app.py before calling these) ─────
def list_all_voice_profiles() -> list:
    return _get("/api/admin/voice_models")


def admin_retrain_voice_model(profile_id: int) -> dict:
    return _post(f"/api/admin/voice_models/{profile_id}/retrain")


def admin_disable_voice_model(profile_id: int) -> dict:
    return _post(f"/api/admin/voice_models/{profile_id}/disable")


def admin_delete_voice_model(profile_id: int) -> dict:
    return _delete(f"/api/admin/voice_models/{profile_id}")


def get_rvc_endpoint() -> dict:
    return _get("/api/rvc_endpoint")


def set_rvc_endpoint(endpoint: str) -> dict:
    return _post("/api/rvc_endpoint", {"endpoint": endpoint})


# ── Notifications: manager-triggered delete/disable events ──────────────────────
def register_webhook(webhook_url: str):
    """Self-registers this app's callback URL so clone-voice-station can push
    delete/disable notifications instead of us having to poll for them. Safe
    to call on every startup — it's just an upsert."""
    _post("/api/webhook", {"webhook_url": webhook_url})


def poll_undelivered_notifications(external_user_id: str) -> list:
    """Fallback path for when the webhook was never registered or a push
    attempt failed — returns notifications the station couldn't deliver yet."""
    try:
        return _get("/api/notifications", params={"external_user_id": external_user_id})
    except VoiceStationError:
        return []


def ack_notification(notification_id: int):
    try:
        _post(f"/api/notifications/{notification_id}/ack")
    except VoiceStationError:
        pass  # best-effort — a missed ack just means it's re-delivered next poll
