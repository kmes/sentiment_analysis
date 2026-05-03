from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from db.database import engine, AsyncSessionLocal

from db.models import Base, InferenceLog, FeedbackLog, ModelLoadLog, ModelEvaluationLog

from contextlib import asynccontextmanager
from fastapi import FastAPI
from datetime import datetime, timedelta, timezone

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


async def get_feedback_disagreement_rate(hours: int = 24) -> float:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    async with AsyncSessionLocal() as session:
        total_stmt = (
            select(func.count())
            .select_from(FeedbackLog)
            .where(FeedbackLog.timestamp >= since)
        )
        total = (await session.execute(total_stmt)).scalar_one()
        if total == 0:
            return 0.0

        disagreement_stmt = (
            select(func.count())
            .select_from(FeedbackLog)
            .join(InferenceLog, FeedbackLog.prediction_id == InferenceLog.prediction_id)
            .where(
                FeedbackLog.timestamp >= since,
                FeedbackLog.true_label != InferenceLog.predicted_label,
            )
        )
        disagreements = (await session.execute(disagreement_stmt)).scalar_one()
        return disagreements / total


async def get_feedback_export(
    model_version: str,
    date_from: datetime | None,
    date_to: datetime | None,
    limit: int | None,
) -> tuple[list[InferenceLog], datetime | None, datetime | None]:
    async with AsyncSessionLocal() as session:
        boundary_stmt = (
            select(func.min(FeedbackLog.timestamp), func.max(FeedbackLog.timestamp))
            .join(InferenceLog, FeedbackLog.prediction_id == InferenceLog.prediction_id)
            .join(ModelLoadLog, InferenceLog.model_load_id == ModelLoadLog.model_load_id)
            .where(ModelLoadLog.model_version == model_version)
        )
        min_ts, max_ts = (await session.execute(boundary_stmt)).one()

        effective_from = max(date_from, min_ts) if date_from and min_ts else min_ts
        effective_to   = min(date_to,   max_ts) if date_to   and max_ts else max_ts

        stmt = (
            select(InferenceLog)
            .join(FeedbackLog, InferenceLog.prediction_id == FeedbackLog.prediction_id)
            .join(ModelLoadLog, InferenceLog.model_load_id == ModelLoadLog.model_load_id)
            .options(selectinload(InferenceLog.feedback))
            .where(ModelLoadLog.model_version == model_version)
            .order_by(FeedbackLog.timestamp.asc())
        )
        if effective_from:
            stmt = stmt.where(FeedbackLog.timestamp >= effective_from)
        if effective_to:
            stmt = stmt.where(FeedbackLog.timestamp <= effective_to)
        if limit:
            stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        return list(result.scalars().all()), effective_from, effective_to


async def insert_model_metrics(payload) -> ModelEvaluationLog:
    async with AsyncSessionLocal() as session:
        record = ModelEvaluationLog(
            model_version=payload.model_version,
            eval_dataset=payload.eval_dataset,
            accuracy=payload.accuracy,
            f1_macro=payload.f1_macro,
            f1_negative=payload.f1_negative,
            f1_neutral=payload.f1_neutral,
            f1_positive=payload.f1_positive,
            eval_loss=payload.eval_loss,
            num_samples=payload.num_samples,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record