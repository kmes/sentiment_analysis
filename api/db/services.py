from db.database import engine, Base, AsyncSessionLocal

from db.models import InferenceLog

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