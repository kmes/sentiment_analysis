from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import analyzer, timer
from middlewares import require_dev_env, check_model_loaded
from schemas import BaseResponse, ModelLabelsResponse, ModelPredictData, ModelPredictResponse, ModelFeedbackData, ModelFeedbackResponse, PredictionsListResponse, PredictionItem

from db import services

import uuid


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

    model_load_id = analyzer.get_model_load_id()

    prediction_id = uuid.uuid4()

    background_tasks.add_task(
        services.save_inference_log_background,
        prediction_id = prediction_id,
        model_load_id = model_load_id,
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
        model_load_id = model_load_id,
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

@router.post("/train")
def model_train():
    return BaseResponse(
        status = "endpoint not available"
    )
