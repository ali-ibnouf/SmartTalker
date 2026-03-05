import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

import sys
from pathlib import Path

# Add project root to PYTHONPATH so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.db.models import Avatar, Customer

async def seed_test_data():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        # Create a test customer first
        c = Customer(
            id="test-customer-1",
            name="Test Customer",
            email="test@example.com",
            api_key="test-key-123"
        )
        session.add(c)
        await session.commit()
    
        # Create a test video avatar that needs migration
        a = Avatar(
            id="test-avatar-1",
            customer_id="test-customer-1",
            name="Old Video Avatar",
            avatar_type="video",
            photo_url="https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=500&q=80",
            photo_preprocessed=False
        )
        session.add(a)
        await session.commit()
        
        print(f"Created Test Avatar: {a.id} (needs migration)")

    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(seed_test_data())
