from fastapi import APIRouter, Depends, HTTPException, status, Query

from middlewares import require_dev_env
from schemas import ModelLoadLogItem, PaginationParams, PaginatedResponse, paginated_response_factory, PredictionItem, PredictionItemResponse

from db import services

import uuid

router = APIRouter(prefix="/logs", dependencies=[Depends(require_dev_env)])

@router.get("/loaded-model")
async def get_model_logs(pagination: PaginationParams = Depends()) -> PaginatedResponse[ModelLoadLogItem]:
    logs, total_items = await services.get_model_load_logs(
        page=pagination.page, 
        limit=pagination.limit
    )
    return paginated_response_factory(
        items=[ModelLoadLogItem.model_validate(l) for l in logs],
        total_items=total_items,
        params=pagination
    )

@router.get("/predictions")
async def get_predictions(
    pagination: PaginationParams = Depends(),
    only_with_feedback: bool | None = Query(None, description="Filter items by feedback presence")
)-> PaginatedResponse[PredictionItem]:
    predictions, total_items = await services.get_all_predictions(
        page=pagination.page, 
        limit=pagination.limit,
        only_with_feedback=only_with_feedback
    )
    return paginated_response_factory(
        items=[PredictionItem.model_validate(p) for p in predictions],
        total_items=total_items,
        params=pagination
    )

@router.get("/predictions/{prediction_id}")
async def model_prediction(prediction_id: uuid.UUID):
    prediction = await services.get_inference_log_by_prediction_id(prediction_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )
    return PredictionItemResponse(
        status = "ok",
        message = "Prediction found",
        prediction =  PredictionItem.model_validate(prediction)
    )