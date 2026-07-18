from fastapi import FastAPI
from src.core.logging_settings import logger 
from contextlib import asynccontextmanager
from sqlalchemy import text
from src.core.database.db import engine
from src.api.endpoints import router as base_router
from src.core.database.models import DocumentChunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("The application is running")
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database connection error:{e}")
    yield 
    await engine.dispose()
    logger.info("Ready")


app = FastAPI(title="production-RAG-service", lifespan=lifespan)

app.include_router(router=base_router)
