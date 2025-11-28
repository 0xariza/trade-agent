"""
Initialize database with tables and seed data
"""
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from backend.db.models import Base
from backend.utils.config import settings

async def init_database():
    """Create all database tables"""
    engine = create_async_engine(settings.database_url)
    
    async with engine.begin() as conn:
        # Drop all tables (careful in production!)
        # await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    asyncio.run(init_database())
