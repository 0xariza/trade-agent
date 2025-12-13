from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import asynccontextmanager
import os
import logging

logger = logging.getLogger(__name__)

def get_database_url() -> str:
    """Get database URL from settings or environment."""
    try:
        from backend.config import settings
        url = settings.database_url
    except Exception:
        # Fallback if settings not available (use SQLite as last resort)
        url = os.getenv(
            "DATABASE_URL",
            "sqlite+aiosqlite:///./alpha_arena.db"  # Only used if settings can't be loaded
        )
    
    # Normalize URL to ensure correct async driver
    url_lower = url.lower()
    if "sqlite" in url_lower:
        # Force SQLite to use aiosqlite driver
        if url_lower.startswith("sqlite+aiosqlite"):
            # Already correct, return as-is
            return url
        elif "://" in url:
            # Extract the path part (everything after ://)
            path_part = url.split("://", 1)[1]
            # Remove any existing sqlite prefix
            if path_part.startswith("/"):
                url = f"sqlite+aiosqlite://{path_part}"
            else:
                url = f"sqlite+aiosqlite:///{path_part}"
        else:
            # No ://, assume it's just a path
            url = f"sqlite+aiosqlite:///{url}"
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        # Force PostgreSQL to use asyncpg (not psycopg2)
        url = url.replace("postgresql://", "postgresql+asyncpg://")
    
    return url

# Get normalized database URL
DATABASE_URL = get_database_url()

# Log the URL (without sensitive info)
safe_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
logger.debug(f"Database URL: {safe_url}")

# Create engine with proper async driver and connection pooling
# IMPORTANT: Always explicitly specify the async driver in the URL
if DATABASE_URL.startswith("sqlite+aiosqlite"):
    # SQLite with aiosqlite - explicitly required for async
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,  # Connection pool size
        max_overflow=10  # Max overflow connections
    )
    logger.info("Using SQLite database (no setup required)")
elif DATABASE_URL.startswith("postgresql+asyncpg"):
    # PostgreSQL with asyncpg - explicitly required for async
    # PERFORMANCE: Connection pooling for better performance
    engine = create_async_engine(
        DATABASE_URL, 
        echo=False,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,  # Connection pool size (default: 5)
        max_overflow=20,  # Max overflow connections (default: 10)
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_timeout=30  # Timeout for getting connection from pool
    )
    logger.info("Using PostgreSQL database with connection pooling")
else:
    # For any other URL, try to create but warn if it might not be async
    if "sqlite" in DATABASE_URL.lower() and "+aiosqlite" not in DATABASE_URL:
        raise ValueError(
            f"SQLite URL must use 'sqlite+aiosqlite://' driver for async. "
            f"Got: {DATABASE_URL}"
        )
    elif "postgresql" in DATABASE_URL.lower() and "+asyncpg" not in DATABASE_URL:
        raise ValueError(
            f"PostgreSQL URL must use 'postgresql+asyncpg://' driver for async. "
            f"Got: {DATABASE_URL}"
        )
    engine = create_async_engine(
        DATABASE_URL, 
        echo=False,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )
    logger.info(f"Using database: {safe_url}")

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def init_db():
    """
    Initialize the database (create tables).
    Handles both SQLite and PostgreSQL.
    """
    try:
        # Import models here to ensure they are registered with Base
        from . import models
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # For SQLite, try to create directory if needed
        if DATABASE_URL.startswith("sqlite"):
            import os
            db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "")
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            # Retry once
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Database initialized successfully (retry)")
                return True
            except Exception as e2:
                logger.error(f"Database initialization failed after retry: {e2}")
                raise
        else:
            raise

@asynccontextmanager
async def get_db():
    """
    Async context manager for getting database session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
