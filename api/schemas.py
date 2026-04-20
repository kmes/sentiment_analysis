from fastapi import Query
from pydantic import BaseModel, field_validator, ValidationError
from datetime import datetime
from typing import Optional, TypeVar, Generic

from dependencies import analyzer

import uuid

class PaginationParams(BaseModel):
    page: int = Query(1, ge=1, description="Page number")
    limit: int = Query(20, ge=1, description="Items per page")

class BaseResponse(BaseModel):
    status: str
    message: str = ""

class BaseListResponse(BaseResponse):
    current_page: int
    total_pages: int
    displayed_items: int
    total_items: int

T = TypeVar("T")
class PaginatedResponse(BaseListResponse, Generic[T]):
    items: list[T]

def paginated_response_factory(items: list[T], total_items: int, params: PaginationParams) -> PaginatedResponse[T]:
    limit = params.limit
    total_pages = (total_items + limit - 1) // limit if limit > 0 else 0
    
    return PaginatedResponse[T](
        status="ok",
        current_page=params.page,
        total_pages=total_pages,
        displayed_items=len(items),
        total_items=total_items,
        items=items
    )

class ModelStatus(BaseModel):
    loaded: bool
    load_id: uuid.UUID | None
    name: str
    version: str

class StatusResponse(BaseResponse):
    model: ModelStatus
    uptime: int

class LoadModelResponse(BaseResponse):
    model: ModelStatus
    time: int

class ModelLabelsResponse(BaseResponse):
    labels: list[str]

class ModelPredictData(BaseModel):
    text: str

class ModelPredictResponse(BaseResponse):
    prediction_id: uuid.UUID
    model_load_id: uuid.UUID
    input_text: str
    predicted_label: str
    confidence: float
    latency_ms: int


class ModelFeedbackData(BaseModel):
    label: str

    @field_validator('label')
    @classmethod
    def label_validator(cls, label: str) -> str:
        if not analyzer.validate_label(label):
            valid_labels = analyzer.get_valid_labels()
            raise ValueError(f"Label must be {', '.join(valid_labels)}")
        return label.lower()
    
    
class ModelFeedbackResponse(BaseResponse):
    prediction_id: uuid.UUID
    label: str


class FeedbackItem(BaseModel):
    true_label: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class PredictionItem(BaseModel):
    prediction_id: uuid.UUID
    model_load_id: uuid.UUID
    timestamp: datetime
    input_text: str
    predicted_label: str
    confidence: float
    latency_ms: int
    feedback: Optional[FeedbackItem] = None

    model_config = {"from_attributes": True}

class PredictionItemResponse(BaseResponse):
    prediction: PredictionItem

class PredictionsListResponse(BaseListResponse):
    predictions: list[PredictionItem]


class ModelLoadLogItem(BaseModel):
    model_load_id: uuid.UUID
    timestamp: datetime
    model_name: str
    model_version: str
    load_time_ms: int

    model_config = {"from_attributes": True}



