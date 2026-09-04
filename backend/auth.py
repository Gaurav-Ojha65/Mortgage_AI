"""
Mortgage AI — Authentication & Role-Based Access Control (RBAC)

Primary product roles:
  - user:          Applicant — submits applications, uploads documents, tracks status
  - loan_officer:  Reviews assigned applications, risk explanations and decisions
  - admin:         Full platform governance, users, audit and analytics

Legacy "underwriter" is accepted as a compatibility role and is treated as a
loan-officer-level reviewer. New users should use the three product roles above.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from database import DATABASE_PATH

SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY and os.environ.get("ENV") == "production":
    raise ValueError("SECRET_KEY environment variable is required in production mode.")
if not SECRET_KEY:
    SECRET_KEY = "development-secret-key-do-not-use-in-prod"

TOKEN_EXPIRE_HOURS = 24
ROLES = ["user", "loan_officer", "admin"]
LEGACY_ROLES = ["underwriter"]
ROLE_HIERARCHY = {"user": 1, "loan_officer": 2, "underwriter": 2, "admin": 3}
security = HTTPBearer(auto_error=False)

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)

class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)
    role: str = Field(..., pattern="^(user|loan_officer|admin)$")
    full_name: str = Field(default="", max_length=100)

class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    full_name: str
    created_at: str
    is_active: bool

def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return hashed.hex(), salt

def verify_password(password: str, hashed: str, salt: str) -> bool:
    check_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(check_hash, hashed)

_token_store = {}

def create_token(user_id: int, username: str, role: str) -> str:
    token = secrets.token_urlsafe(48)
    _token_store[token] = {
        "user_id": user_id, "username": username, "role": role,
        "full_name": "", "expires": datetime.now() + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "created_at": datetime.now().isoformat(),
    }
    now = datetime.now()
    for t, d in list(_token_store.items()):
        if d["expires"] < now:
            del _token_store[t]
    return token

def validate_token(token: str) -> Optional[dict]:
    data = _token_store.get(token)
    if not data:
        return None
    if datetime.now() > data["expires"]:
        del _token_store[token]
        return None
    return data

def revoke_token(token: str):
    _token_store.pop(token, None)

_attempt_store = {}

def check_lockout(username: str) -> tuple[bool, str]:
    data = _attempt_store.get(username)
    if not data:
        return False, ""
    now = datetime.now()
    if data["lockout_until"] and now < data["lockout_until"]:
        mins_left = int((data["lockout_until"] - now).total_seconds() / 60) + 1
        return True, f"Too many failed attempts. Account locked for {mins_left} more minutes."
    if data["lockout_until"] and now >= data["lockout_until"]:
        del _attempt_store[username]
    return False, ""

def record_failed_attempt(username: str):
    now = datetime.now()
    data = _attempt_store.get(username, {"count": 0, "lockout_until": None})
    data["count"] += 1
    if data["count"] >= 5:
        data["lockout_until"] = now + timedelta(minutes=15)
    _attempt_store[username] = data

def clear_attempts(username: str):
    _attempt_store.pop(username, None)

def init_users_table():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            full_name TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    """)
    conn.commit()
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        seed_users = [
            ("admin", "admin123", "admin", "System Administrator"),
            ("officer", "lo2024", "loan_officer", "Loan Officer"),
            ("user", "user2024", "user", "Demo Applicant"),
        ]
        for username, password, role, full_name in seed_users:
            pw_hash, salt = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, password_salt, role, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (username, pw_hash, salt, role, full_name, datetime.now().isoformat()),
            )
        conn.commit()
    conn.close()

def authenticate_user(username: str, password: str) -> dict:
    is_locked, msg = check_lockout(username)
    if is_locked:
        raise HTTPException(status_code=429, detail=msg)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND is_active = 1", (username,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        record_failed_attempt(username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    if not verify_password(password, user["password_hash"], user["password_salt"]):
        record_failed_attempt(username)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    clear_attempts(username)
    return dict(user)

def get_all_users() -> list:
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, username, role, full_name, is_active, created_at, last_login FROM users ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_user_db(data: UserCreate) -> dict:
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    pw_hash, salt = hash_password(data.password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, password_salt, role, full_name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (data.username, pw_hash, salt, data.role, data.full_name, datetime.now().isoformat()),
        )
        conn.commit()
        user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="Username already exists")
    conn.close()
    return {"id": user_id, "username": data.username, "role": data.role, "full_name": data.full_name}

def update_last_login(user_id: int):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    user_data = validate_token(credentials.credentials)
    if user_data is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_data

async def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[dict]:
    if credentials is None:
        return None
    return validate_token(credentials.credentials)

def require_role(*allowed_roles):
    async def role_checker(user: dict = Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(status_code=403, detail=f"Access denied. Required role: {', '.join(allowed_roles)}. Your role: {user['role']}")
        return user
    return role_checker

def require_min_role(min_role: str):
    async def role_checker(user: dict = Depends(get_current_user)):
        if ROLE_HIERARCHY.get(user["role"], 0) < ROLE_HIERARCHY.get(min_role, 0):
            raise HTTPException(status_code=403, detail=f"Access denied. Minimum role required: {min_role}. Your role: {user['role']}")
        return user
    return role_checker
