from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from db.database import engine, AsyncSessionLocal

from db.models import Base, InferenceLog, FeedbackLog

from contextlib import asynccontextmanager
from fastapi import FastAPI

import uuid


# Startup: crea tabelle se non esistono
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # idempotente: salta tabelle già esistenti
    yield


async def save_inference_log_background(
    prediction_id: uuid.UUID,
    model_version: str,
    input_text: str,
    predicted_label: str,
    confidence: float,
    latency_ms: int,
) -> None:
    async with AsyncSessionLocal() as session:
        log = InferenceLog(
            prediction_id=prediction_id,
            model_version=model_version,
            input_text=input_text,
            predicted_label=predicted_label,
            confidence=confidence,
            latency_ms=latency_ms,
        )
        session.add(log)
        await session.commit()

async def get_inference_log_by_prediction_id(
    session: AsyncSession,
    prediction_id: uuid.UUID,
) -> InferenceLog | None:
    result = await session.execute(
        select(InferenceLog)
        .where(InferenceLog.prediction_id == prediction_id)
        .options(selectinload(InferenceLog.feedback))
    )
    return result.scalar_one_or_none()

async def create_feedback(
    session: AsyncSession,
    prediction_id: uuid.UUID,
    true_label: str,
) -> FeedbackLog:
    feedback = FeedbackLog(
        prediction_id=prediction_id,
        true_label=true_label
    )
    session.add(feedback)
    await session.flush()
    return feedback