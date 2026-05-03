from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from typing import Optional

from dependencies import (
    analyzer, timer,
    prediction_latency_ms, predictions_total,
    prediction_text_length, prediction_confidence,
    feedback_total,
    model_accuracy, model_f1_macro, model_f1_per_label,
)
from middlewares import check_model_loaded
from schemas import (
    BaseResponse, ModelLabelsResponse, ModelPredictData, ModelPredictResponse,
    ModelFeedbackData, ModelFeedbackResponse,
    FeedbackExportRecord, FeedbackExportResponse,
    ModelMetricsPayload, ModelMetricsResponse,
)

from db import services

import uuid


router = APIRouter(prefix="/model", dependencies=[Depends(check_model_loaded)])

# Router senza check_model_loaded per endpoint di data management
data_router = APIRouter(prefix="/model")


@router.get("/labels")
def model_labels():
    return ModelLabelsResponse(
        status="ok",
        labels=analyzer.get_valid_labels()
    )


@router.post("/predictions/{prediction_id}/feedback")
async def model_feedback(prediction_id: uuid.UUID, data: ModelFeedbackData):
    inference_log = await services.get_inference_log_by_prediction_id(prediction_id)
    if inference_log is not None and inference_log.feedback is None:
        await services.create_feedback(
            prediction_id=prediction_id,
            true_label=data.label
        )
        feedback_total.labels(true_label=data.label).inc()

    return ModelFeedbackResponse(
        status="feedback received",
        message="Thank you for you feedback",
        prediction_id=prediction_id,
        label=data.label
    )


@router.post("/predict")
def model_predict(data: ModelPredictData, background_tasks: BackgroundTasks):

    timer.reset_timer()
    label, score = analyzer.predict(data.text)
    prediction_time = timer.partial_timer()

    model_load_id = analyzer.get_model_load_id()
    prediction_id = uuid.uuid4()

    background_tasks.add_task(
        services.save_inference_log_background,
        prediction_id=prediction_id,
        model_load_id=model_load_id,
        input_text=data.text,
        predicted_label=label,
        confidence=score,
        latency_ms=prediction_time,
    )

    prediction_latency_ms.observe(prediction_time)
    predictions_total.labels(predicted_label=label).inc()
    prediction_text_length.observe(len(data.text))
    prediction_confidence.observe(score)

    return ModelPredictResponse(
        status="predicted",
        message=f"Result {label} with {score:.2f} of confidence",
        input_text=data.text,
        prediction_id=prediction_id,
        model_load_id=model_load_id,
        predicted_label=label,
        confidence=score,
        latency_ms=prediction_time
    )


@router.post("/train")
def model_train():
    return BaseResponse(
        status="endpoint not available"
    )


@data_router.get("/feedback-export", response_model=FeedbackExportResponse)
async def feedback_export(
    model_version: str = Query(...),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: Optional[int] = Query(None, ge=1),
):
    records, effective_from, effective_to = await services.get_feedback_export(
        model_version=model_version,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )

    export_records = [
        FeedbackExportRecord(
            prediction_id=r.prediction_id,
            input_text=r.input_text,
            predicted_label=r.predicted_label,
            confidence=r.confidence,
            true_label=r.feedback.true_label,
            predicted_at=r.timestamp,
            feedback_at=r.feedback.timestamp,
        )
        for r in records
    ]

    return FeedbackExportResponse(
        model_version=model_version,
        exported_at=datetime.now(timezone.utc),
        date_from_requested=date_from,
        date_from_effective=effective_from,
        date_to_requested=date_to,
        date_to_effective=effective_to,
        total_records=len(export_records),
        limit_applied=limit,
        records=export_records,
    )


@data_router.post("/metrics", response_model=ModelMetricsResponse, status_code=status.HTTP_201_CREATED)
async def post_model_metrics(payload: ModelMetricsPayload):
    record = await services.insert_model_metrics(payload)

    model_accuracy.labels(model_version=payload.model_version).set(payload.accuracy)
    model_f1_macro.labels(model_version=payload.model_version).set(payload.f1_macro)
    model_f1_per_label.labels(label="negative", model_version=payload.model_version).set(payload.f1_negative)
    model_f1_per_label.labels(label="neutral",  model_version=payload.model_version).set(payload.f1_neutral)
    model_f1_per_label.labels(label="positive", model_version=payload.model_version).set(payload.f1_positive)

    return record
