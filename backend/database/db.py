from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import asynccontextmanager
import os

# Database URL
# Use environment variable with fallback for development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://alpha_user:alpha_password@localhost:5432/alpha_arena"
)

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def init_db():
    """
    Initialize the database (create tables).
    """
    # Import models here to ensure they are registered with Base
    from . import models
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
