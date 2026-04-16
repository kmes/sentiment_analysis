from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import analyzer, timer
from schemas import BaseResponse, ModelLabelsResponse, ModelPredictData, ModelPredictResponse, ModelFeedbackData, ModelFeedbackResponse, PredictionsListResponse, PredictionItem

from db import services
from db.database import get_db

import os

import uuid

def require_dev_env():
    app_env = os.environ.get("APP_ENV", "prod")
    if app_env.lower() != "dev":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is available only in the development environment."
        )

def check_model_loaded():
    if not analyzer.model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please load the model before making requests."
        )

router = APIRouter(prefix="/model", dependencies=[Depends(check_model_loaded)])

@router.get("/labels")
def model_labels():
    return ModelLabelsResponse(
        status = "ok",
        labels = analyzer.get_valid_labels()
    )
    
@router.post("/predict")
def model_predict(data: ModelPredictData, background_tasks: BackgroundTasks):

    timer.reset_timer()
    label, score = analyzer.predict(data.text)
    prediction_time = timer.partial_timer()

    prediction_id = uuid.uuid4()

    background_tasks.add_task(
        services.save_inference_log_background,
        prediction_id = prediction_id,
        model_version = "v1.0",
        input_text = data.text,
        predicted_label = label,
        confidence = score,
        latency_ms = prediction_time,
    )

    return ModelPredictResponse(
        status = "predicted",
        message = f"Result {label} with {score:.2f} of confidence",
        text = data.text,
        prediction_id = prediction_id,
        label = label,
        score = score,
        prediction_time = prediction_time
    )

@router.post("/feedback")
async def model_feedback(data: ModelFeedbackData):
    # Il controllo che data.label sia valido viene fatto nel validator di ModelFeedbackData

    inference_log = await services.get_inference_log_by_prediction_id(data.prediction_id)
    # Controllo se il prediction_id esiste e se non ha già un feedback associato
    if inference_log is not None and inference_log.feedback is None:
        # Salvo il feedback solo se il prediction_id è valido e non ha già un feedback associato
        await services.create_feedback(
            prediction_id = data.prediction_id,
            true_label = data.label
        )

    # Ritorno un messaggio di ringraziamento per il feedback, 
    # indipendentemente dal fatto che il prediction_id sia valido o meno
    return ModelFeedbackResponse(
        status = "feedback accepted",
        message = "Thank you for you feedback",
        prediction_id = data.prediction_id,
        label = data.label
    )


@router.get("/predictions", dependencies=[Depends(require_dev_env)])
async def get_predictions(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, description="Items per page")
):
    predictions, total_items = await services.get_all_predictions(page=page, limit=limit)
    total_pages = (total_items + limit - 1) // limit if limit > 0 else 0
    return PredictionsListResponse(
        status = "ok",
        current_page = page,
        total_pages = total_pages,
        displayed_items = len(predictions),
        total_items = total_items,
        predictions = [PredictionItem.model_validate(p) for p in predictions]
    )


@router.post("/train")
def model_train():
    return BaseResponse(
        status = "endpoint not available"
    )
