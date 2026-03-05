#!/usr/bin/env python3
"""Migrate existing Video avatars to the new RunPod face preprocessing pipeline.

This script scans all avatars of type 'video' where photo_preprocessed is False
and they have a photo_url. It connects to the configured RunPod Serverless API,
runs face preprocessing on the photo, and saves the resulting face_data_url.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to PYTHONPATH so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from src.config import Settings
from src.db.models import Avatar
from src.services.runpod_client import RunPodServerless, RunPodError

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("migration")

async def run_migration():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    settings = Settings(_env_file=env_path)
    
    # 1. Initialize Database
    try:
        engine = create_async_engine(settings.database_url, echo=False)
        logger.info(f"Connected to DB: {settings.database_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Database: {e}")
        return

    # 2. Initialize RunPod Client
    try:
        runpod_client = RunPodServerless(settings)
        if not runpod_client._endpoint_preprocess:
            logger.error("RUNPOD_ENDPOINT_PREPROCESS is not set in the configuration.")
            return
        logger.info(f"Initialized RunPod client: {runpod_client._endpoint_preprocess}")
    except Exception as e:
        logger.error(f"Failed to initialize RunPod client: {e}")
        return

    success_count = 0
    failure_count = 0

    from sqlalchemy.orm import sessionmaker
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        # Load avatars needing migration
        async with async_session() as session:
            stmt = select(Avatar).where(
                Avatar.avatar_type == "video",
                Avatar.photo_preprocessed == False,
                Avatar.photo_url != ""
            )
            result = await session.execute(stmt)
            avatars = result.scalars().all()

        if not avatars:
            logger.info("No video avatars found that require migration.")
            return
        
        logger.info(f"Found {len(avatars)} video avatars requiring preprocessing.")

        for avatar in avatars:
            logger.info(f"Migrating avatar: ({avatar.id})")
            logger.info(f"  Photo URL: {avatar.photo_url}")
            
            try:
                # Issue preprocessing job to RunPod
                rp_result = await runpod_client.preprocess_face(
                    photo_url=avatar.photo_url,
                    employee_id=avatar.id
                )
                
                # Update DB
                async with engine.begin() as conn:
                    await conn.execute(
                        update(Avatar)
                        .where(Avatar.id == avatar.id)
                        .values(
                            face_data_url=rp_result.face_data_url,
                            photo_preprocessed=True
                        )
                    )
                
                logger.info(f"  [OK] Successfully processed! Cost: ${rp_result.cost_usd:.5f}")
                logger.info(f"  Face Data URL: {rp_result.face_data_url}")
                success_count += 1
                
            except RunPodError as rp_err:
                logger.error(f"  [ERROR] RunPod processing failed for {avatar.id}: {rp_err}")
                failure_count += 1
            except Exception as e:
                logger.error(f"  [ERROR] Unhandled exception processing {avatar.id}: {e}")
                failure_count += 1

    finally:
        await runpod_client.close()
        await engine.dispose()
        
        logger.info("-" * 40)
        logger.info("Migration Summary:")
        logger.info(f"  Total processed: {success_count + failure_count}")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Failed:  {failure_count}")

if __name__ == "__main__":
    try:
        asyncio.run(run_migration())
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user.")
