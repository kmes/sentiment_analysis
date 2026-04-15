from pydantic import BaseModel, field_validator, ValidationError

from dependencies import analyzer

import uuid

class BaseResponse(BaseModel):
    status: str
    message: str = ""

class StatusResponse(BaseResponse):
    model: str
    uptime: int

class LoadModelResponse(BaseResponse):
    time: int

class ModelLabelsResponse(BaseResponse):
    labels: list[str]

class ModelPredictData(BaseModel):
    text: str

class ModelPredictResponse(BaseResponse):
    prediction_id: uuid.UUID
    text: str
    label: str
    score: float
    prediction_time: int


class ModelFeedbackData(BaseModel):
    prediction_id: uuid.UUID
    label: str

    @field_validator('prediction_id')
    @classmethod
    def prediction_id_validator(cls, prediction_id: uuid.UUID) -> uuid.UUID:
        return prediction_id

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