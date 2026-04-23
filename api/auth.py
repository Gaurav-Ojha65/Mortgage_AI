"""
JWT Authentication and Authorization for Mortgage AI API
Supports role-based access control (admin, analyst, auditor)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
from enum import Enum

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


class Role(str, Enum):
    """User roles."""
    ADMIN = "admin"
    ANALYST = "analyst"
    AUDITOR = "auditor"


class User(BaseModel):
    """User model."""
    username: str
    email: str
    role: Role
    disabled: bool = False


class UserInDB(User):
    """User with password hash."""
    hashed_password: str


class Token(BaseModel):
    """Token response."""
    access_token: str
    token_type: str
    expires_in: int


class TokenData(BaseModel):
    """Token payload."""
    username: Optional[str] = None
    role: Optional[str] = None


# Mock database - replace with real database in production
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@mortgage-ai.com",
        "hashed_password": pwd_context.hash("admin123"),
        "role": Role.ADMIN,
        "disabled": False
    },
    "analyst": {
        "username": "analyst",
        "email": "analyst@mortgage-ai.com",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": Role.ANALYST,
        "disabled": False
    },
    "auditor": {
        "username": "auditor",
        "email": "auditor@mortgage-ai.com",
        "hashed_password": pwd_context.hash("auditor123"),
        "role": Role.AUDITOR,
        "disabled": False
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)


def get_user(db: Dict, username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(db: Dict, username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user."""
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        role: str = payload.get("role")
        token_type: str = payload.get("type")

        if username is None or token_type != "access":
            raise credentials_exception

        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception

    user = get_user(fake_users_db, token_data.username)
    if user is None:
        raise credentials_exception
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")

    return User(
        username=user.username,
        email=user.email,
        role=user.role,
        disabled=user.disabled
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(allowed_roles: List[Role]):
    """Require specific role(s)."""
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in [r.value for r in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


# Role dependencies
require_admin = require_role([Role.ADMIN])
require_analyst = require_role([Role.ADMIN, Role.ANALYST])
require_auditor = require_role([Role.ADMIN, Role.AUDITOR])


class AuditLogger:
    """Audit logging for compliance."""

    def __init__(self):
        self.logs: List[Dict] = []

    def log(
        self,
        action: str,
        user: str,
        details: Optional[Dict] = None,
        ip: Optional[str] = None
    ):
        """Log an audit event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "ip": ip,
            "details": details or {}
        }
        self.logs.append(entry)
        # In production: write to database or log aggregation system
        print(f"[AUDIT] {entry}")

    def get_logs(
        self,
        user: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query audit logs."""
        filtered = self.logs

        if user:
            filtered = [l for l in filtered if l["user"] == user]
        if action:
            filtered = [l for l in filtered if l["action"] == action]
        if start_date:
            filtered = [
                l for l in filtered
                if datetime.fromisoformat(l["timestamp"]) >= start_date
            ]
        if end_date:
            filtered = [
                l for l in filtered
                if datetime.fromisoformat(l["timestamp"]) <= end_date
            ]

        return filtered[-limit:]


# Global audit logger instance
audit_logger = AuditLogger()
