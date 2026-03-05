import asyncio
from sqlalchemy.ext.asyncio import create_async_engine

import sys
from pathlib import Path

# Add project root to PYTHONPATH so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.db.models import Base

async def init_db():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=True)
    
    async with engine.begin() as conn:
        print("Creating all tables from src.db.models...")
        await conn.run_sync(Base.metadata.create_all)
        print("Done.")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())
