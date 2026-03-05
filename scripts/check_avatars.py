import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

# Set up local paths to import src
import sys
sys.path.insert(0, r"C:\Users\User\Documents\smart talker\SmartTalker")

from src.config import get_settings
from src.db.models import Avatar

async def main():
    settings = get_settings()
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        result = await session.execute(select(Avatar))
        avatars = result.scalars().all()
        
        print(f"Total Avatars: {len(avatars)}")
        for a in avatars:
            print(f"- ID: {a.id}, Name: {a.name}, Type: {a.avatar_type}, Photo: {a.photo_url}, Preprocessed: {a.photo_preprocessed}")

if __name__ == "__main__":
    asyncio.run(main())
