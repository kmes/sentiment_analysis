from fastapi import APIRouter, Depends, BackgroundTasks

from ..boostrap import analyzer, timer
from ..schemas import BaseResponse, ModelLabelsResponse, ModelPredictData, ModelPredictResponse, ModelFeedbackData, ModelFeedbackResponse

from ..db import services

import uuid

async def check_model_loaded():
    if not analyzer.model_loaded():
        return BaseResponse(
            status = "Model not loaded",
            message = "Please load model before"
        )
    else:
        yield

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
        msg = f"Result {label} with {score:.2f} of confidence",
        prediction_id = prediction_id,
        label = label,
        score = score,
        prediction_time = prediction_time
    )

@router.post("/feedback")
def model_feedback(data: ModelFeedbackData):
    # Todo:
    ## check prediction_id

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