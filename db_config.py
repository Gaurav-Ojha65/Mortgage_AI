"""
Database Configuration for Mortgage AI

Provides unified database configuration supporting both PostgreSQL (production)
and SQLite (development/fallback). Includes connection pooling, retry logic,
and works with both SQLAlchemy sync and async engines.

Usage:
    from db_config import get_database_url, create_engine, create_async_engine

    # Get database URL (reads from env, falls back to SQLite)
    db_url = get_database_url()

    # Create sync engine with pooling
    engine = create_engine(db_url)

    # Create async engine
    async_engine = create_async_engine(db_url)
"""

import os
import time
import logging
from typing import Optional, Tuple
from functools import wraps

from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Default SQLite path for development/fallback
DEFAULT_SQLITE_PATH = "mortgage.db"

# Connection pool settings
POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 10
POOL_OVERFLOW = 20
POOL_TIMEOUT = 30

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds

# Database URL format templates
POSTGRESQL_TEMPLATE = "postgresql://{user}:{password}@{host}:{port}/{database}"
POSTGRESQL_ASYNC_TEMPLATE = "postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
SQLITE_TEMPLATE = "sqlite:///{path}"
SQLITE_ASYNC_TEMPLATE = "sqlite+aiosqlite:///{path}"


# =============================================================================
# Database URL Functions
# =============================================================================


def get_database_url(
    use_async: bool = False,
    prefer_sqlite: bool = False
) -> str:
    """
    Get database URL from environment or fallback to SQLite.

    Args:
        use_async: If True, return async-compatible URL
        prefer_sqlite: If True, always use SQLite (for local dev)

    Returns:
        Database URL string
    """
    # Force SQLite for local development
    if prefer_sqlite or os.getenv("DB_USE_SQLITE", "").lower() == "true":
        sqlite_path = os.getenv("SQLITE_PATH", DEFAULT_SQLITE_PATH)
        if use_async:
            return SQLITE_ASYNC_TEMPLATE.format(path=sqlite_path)
        return SQLITE_TEMPLATE.format(path=sqlite_path)

    # Try to build PostgreSQL URL from environment
    pg_user = os.getenv("POSTGRES_USER", os.getenv("DB_USER", "postgres"))
    pg_password = os.getenv("POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "postgres"))
    pg_host = os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost"))
    pg_port = os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))
    pg_database = os.getenv("POSTGRES_DB", os.getenv("DB_NAME", "mortgage"))

    # Check if DATABASE_URL is provided directly
    direct_url = os.getenv("DATABASE_URL")
    if direct_url:
        if use_async and "postgresql://" in direct_url:
            return direct_url.replace("postgresql://", "postgresql+asyncpg://")
        return direct_url

    # Build PostgreSQL URL from components
    try:
        if use_async:
            return POSTGRESQL_ASYNC_TEMPLATE.format(
                user=pg_user,
                password=pg_password,
                host=pg_host,
                port=pg_port,
                database=pg_database
            )
        return POSTGRESQL_TEMPLATE.format(
            user=pg_user,
            password=pg_password,
            host=pg_host,
            port=pg_port,
            database=pg_database
        )
    except KeyError as e:
        logger.warning(f"Missing PostgreSQL config: {e}. Falling back to SQLite.")
        return get_database_url(use_async=use_async, prefer_sqlite=True)


def is_postgresql_url(url: str) -> bool:
    """Check if URL is PostgreSQL (not SQLite)."""
    return url.startswith("postgresql://") or url.startswith("postgresql+asyncpg://")


def get_db_type(url: str) -> str:
    """Get database type from URL."""
    if url.startswith("sqlite+aiosqlite://"):
        return "sqlite_async"
    elif url.startswith("sqlite://"):
        return "sqlite"
    elif url.startswith("postgresql+asyncpg://"):
        return "postgresql_async"
    elif url.startswith("postgresql://"):
        return "postgresql"
    return "unknown"


# =============================================================================
# Engine Creation Functions
# =============================================================================


def create_engine(
    url: Optional[str] = None,
    pool_size: int = POOL_MAX_SIZE,
    pool_min_size: int = POOL_MIN_SIZE,
    echo: bool = False,
    **kwargs
) -> Engine:
    """
    Create SQLAlchemy sync engine with connection pooling.

    Args:
        url: Database URL (auto-detected if None)
        pool_size: Maximum pool size
        pool_min_size: Minimum pool size
        echo: Enable SQL echo for debugging
        **kwargs: Additional arguments passed to create_engine

    Returns:
        SQLAlchemy Engine instance
    """
    if url is None:
        url = get_database_url()

    db_type = get_db_type(url)
    logger.info(f"Creating sync engine for {db_type}")

    # Configure pool settings for PostgreSQL
    if is_postgresql_url(url):
        engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=POOL_OVERFLOW,
            pool_timeout=POOL_TIMEOUT,
            pool_pre_ping=True,  # Enable connection health checks
            echo=echo,
            **kwargs
        )
    else:
        # SQLite doesn't need pooling
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=echo,
            **kwargs
        )

    # Register event listeners
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_conn, connection_record):
        logger.debug("New database connection established")

    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")

    return engine


def create_async_engine(
    url: Optional[str] = None,
    echo: bool = False,
    **kwargs
) -> AsyncEngine:
    """
    Create SQLAlchemy async engine.

    Args:
        url: Database URL (auto-detected if None)
        echo: Enable SQL echo for debugging
        **kwargs: Additional arguments passed to create_async_engine

    Returns:
        SQLAlchemy AsyncEngine instance
    """
    if url is None:
        url = get_database_url(use_async=True)

    db_type = get_db_type(url)
    logger.info(f"Creating async engine for {db_type}")

    if is_postgresql_url(url):
        # PostgreSQL with asyncpg
        engine = create_async_engine(
            url,
            echo=echo,
            pool_pre_ping=True,
            **kwargs
        )
    else:
        # SQLite with aiosqlite
        engine = create_async_engine(
            url,
            echo=echo,
            **kwargs
        )

    return engine


# =============================================================================
# Connection Testing with Retry
# =============================================================================


def test_connection(
    engine: Engine,
    max_retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF
) -> Tuple[bool, str]:
    """
    Test database connection with retry logic.

    Args:
        engine: SQLAlchemy Engine to test
        max_retries: Maximum number of retry attempts
        backoff: Initial backoff in seconds (doubles each retry)

    Returns:
        Tuple of (success: bool, message: str)
    """
    current_backoff = backoff

    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            return True, f"Connection successful on attempt {attempt}"
        except Exception as e:
            error_msg = f"Connection attempt {attempt} failed: {str(e)}"
            logger.warning(error_msg)

            if attempt < max_retries:
                logger.info(f"Retrying in {current_backoff:.1f} seconds...")
                time.sleep(current_backoff)
                current_backoff *= 2
            else:
                return False, f"All {max_retries} attempts failed: {str(e)}"

    return False, "Unknown error"


async def test_async_connection(
    engine: AsyncEngine,
    max_retries: int = MAX_RETRIES,
    backoff: float = RETRY_BACKOFF
) -> Tuple[bool, str]:
    """
    Test async database connection with retry logic.

    Args:
        engine: SQLAlchemy AsyncEngine to test
        max_retries: Maximum number of retry attempts
        backoff: Initial backoff in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select

    current_backoff = backoff

    for attempt in range(1, max_retries + 1):
        try:
            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
            return True, f"Connection successful on attempt {attempt}"
        except Exception as e:
            error_msg = f"Connection attempt {attempt} failed: {str(e)}"
            logger.warning(error_msg)

            if attempt < max_retries:
                logger.info(f"Retrying in {current_backoff:.1f} seconds...")
                time.sleep(current_backoff)
                current_backoff *= 2
            else:
                return False, f"All {max_retries} attempts failed: {str(e)}"

    return False, "Unknown error"


# =============================================================================
# Session Factory Creation
# =============================================================================


def create_session_factory(engine: Engine):
    """
    Create SQLAlchemy session factory for sync engine.

    Args:
        engine: SQLAlchemy Engine

    Returns:
        Session factory (call to get session)
    """
    from sqlalchemy.orm import sessionmaker, scoped_session

    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )

    # Thread-safe scoped session
    return scoped_session(session_factory)


def create_async_session_factory(engine: AsyncEngine):
    """
    Create SQLAlchemy async session factory.

    Args:
        engine: SQLAlchemy AsyncEngine

    Returns:
        Async session factory
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )


# =============================================================================
# Dependency Injection Helpers for FastAPI
# =============================================================================


def get_db_session(engine: Engine):
    """
    FastAPI dependency for getting database session.

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db_session(engine))):
            ...
    """
    session = create_session_factory(engine)()
    try:
        yield session
    finally:
        session.close()


async def get_async_db_session(engine: AsyncEngine):
    """
    FastAPI dependency for getting async database session.

    Usage:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_async_db_session(engine))):
            ...
    """
    session_factory = create_async_session_factory(engine)
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


# =============================================================================
# Schema Initialization
# =============================================================================


def init_sqlite_schema(engine: Engine):
    """
    Initialize SQLite schema for development/testing.

    Creates tables matching the PostgreSQL schema.
    """
    from sqlalchemy import MetaData, Table, Column, Integer, String, Float, DateTime, Text

    metadata = MetaData()

    # Decisions table
    Table('decisions', metadata,
        Column('id', Integer, primary_key=True),
        Column('timestamp', String, nullable=False),
        Column('income', Float, nullable=False),
        Column('loan_amount', Float, nullable=False),
        Column('credit_score', Integer, nullable=False),
        Column('decision', String, nullable=False),
        Column('risk_level', String, nullable=False),
        Column('default_probability', Float),
        Column('emi', Float, nullable=False),
        Column('advice', Text),
    )

    # Audit logs table
    Table('audit_logs', metadata,
        Column('id', Integer, primary_key=True),
        Column('timestamp', String, nullable=False),
        Column('user_id', String),
        Column('action', String, nullable=False),
        Column('details', Text),
        Column('ip_address', String),
    )

    # Users table
    Table('users', metadata,
        Column('id', Integer, primary_key=True),
        Column('username', String, unique=True, nullable=False),
        Column('hashed_password', String, nullable=False),
        Column('role', String, nullable=False),
        Column('created_at', String),
    )

    metadata.create_all(engine)
    logger.info("SQLite schema initialized")


# =============================================================================
# Main entry point for testing
# =============================================================================


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test sync connection
    print("\n=== Testing Sync Connection ===")
    url = get_database_url()
    print(f"Database URL: {url}")
    print(f"Database type: {get_db_type(url)}")

    engine = create_engine(url)
    success, message = test_connection(engine)
    print(f"Connection test: {message}")

    if success:
        print("\n=== Testing Session Factory ===")
        Session = create_session_factory(engine)
        session = Session()
        print(f"Session created: {session}")
        session.close()
        print("Session closed successfully")

    # Test async connection
    print("\n=== Testing Async Connection ===")
    async_url = get_database_url(use_async=True)
    print(f"Async URL: {async_url}")

    async_engine = create_async_engine(async_url)
    import asyncio

    async def test_async():
        success, message = await test_async_connection(async_engine)
        print(f"Async connection test: {message}")
        await async_engine.dispose()

    asyncio.run(test_async())

    print("\n=== All tests complete ===")
