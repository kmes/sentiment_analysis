from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from db.database import engine, AsyncSessionLocal

from db.models import Base, InferenceLog, FeedbackLog, ModelLoadLog

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
    model_load_id: uuid.UUID,
    input_text: str,
    predicted_label: str,
    confidence: float,
    latency_ms: int,
) -> None:
    async with AsyncSessionLocal() as session:
        log = InferenceLog(
            prediction_id=prediction_id,
            model_load_id=model_load_id,
            input_text=input_text,
            predicted_label=predicted_label,
            confidence=confidence,
            latency_ms=latency_ms,
        )
        session.add(log)
        await session.commit()

async def save_model_load_log_background(
    model_load_id: uuid.UUID,
    model_name: str,
    model_version: str,
    load_time_ms: int,
) -> None:
    async with AsyncSessionLocal() as session:
        log = ModelLoadLog(
            model_load_id=model_load_id,
            model_name=model_name,
            model_version=model_version,
            load_time_ms=load_time_ms,
        )
        session.add(log)
        await session.commit()

async def get_inference_log_by_prediction_id(
    prediction_id: uuid.UUID,
) -> InferenceLog | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(InferenceLog)
            .where(InferenceLog.prediction_id == prediction_id)
            .options(selectinload(InferenceLog.feedback))
        )
        return result.scalar_one_or_none()

async def create_feedback(
    prediction_id: uuid.UUID,
    true_label: str,
) -> FeedbackLog:
    async with AsyncSessionLocal() as session:
        feedback = FeedbackLog(
            prediction_id=prediction_id,
            true_label=true_label
        )
        session.add(feedback)
        await session.commit()
        return feedback


async def get_all_predictions(page: int = 1, limit: int = 20, only_with_feedback: bool | None = None) -> tuple[list[InferenceLog], int]:
    async with AsyncSessionLocal() as session:
        count_stmt = select(func.count()).select_from(InferenceLog)
        stmt = select(InferenceLog).options(selectinload(InferenceLog.feedback))

        if only_with_feedback is True:
            count_stmt = count_stmt.where(InferenceLog.feedback.has())
            stmt = stmt.where(InferenceLog.feedback.has())

        count_result = await session.execute(count_stmt)
        total_items = count_result.scalar_one()

        offset = (page - 1) * limit
        stmt = stmt.order_by(InferenceLog.timestamp.desc()).offset(offset).limit(limit)

        result = await session.execute(stmt)
        return list(result.scalars().all()), total_items


async def get_model_load_logs(page: int = 1, limit: int = 20) -> tuple[list[ModelLoadLog], int]:
    async with AsyncSessionLocal() as session:
        count_stmt = select(func.count()).select_from(ModelLoadLog)
        
        count_result = await session.execute(count_stmt)
        total_items = count_result.scalar_one()

        stmt = select(ModelLoadLog)
        offset = (page - 1) * limit
        stmt = stmt.order_by(ModelLoadLog.timestamp.desc()).offset(offset).limit(limit)

        result = await session.execute(stmt)
        return list(result.scalars().all()), total_items