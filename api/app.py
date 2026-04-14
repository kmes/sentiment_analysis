from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel, field_validator, ValidationError
# from typing import Optional
from SentimentAnalyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer("cardiffnlp/twitter-roberta-base-sentiment-latest")


app = FastAPI()

class BaseResponse(BaseModel):
    status: str
    message: str = ""

class StatusResponse(BaseResponse):
    model: str

@app.get("/status")
def status_endopoint():
    return StatusResponse(
        status = "ok",
        model = ("" if analyzer.model_loaded() else "not ") + "loaded"
    )

@app.get("/load-model")
def load_model():
    if analyzer.model_loaded():
        return BaseResponse(
            status = "already loaded"
        )
    else:
        analyzer.load_model()
        return BaseResponse(
            status = "model " + "loaded!" if analyzer.model_loaded() else "not loaded"
        )


async def check_model_loaded():
    if not analyzer.model_loaded():
        return BaseResponse(
            status = "Model not loaded",
            message = "Please load model before"
        )
    else:
        yield


model_router = APIRouter(prefix="/model", dependencies=[Depends(check_model_loaded)])
app.include_router(model_router)

class ModelLabelsResponse(BaseResponse):
    labels: list[str]

@model_router.get("/labels")
def model_labels():
    return ModelLabelsResponse(
        status = "ok",
        labels = analyzer.get_valid_labels()
    )


class ModelPredictData(BaseModel):
    text: str

class ModelPredictResponse(BaseResponse):
    prediction_id: str
    label: str
    score: float
    time: int
    
@model_router.post("/predict")
def model_predict(data: ModelPredictData):
    label, score = analyzer.predict(data.text)

    prediction_id = 0

    return ModelLabelsResponse(
        status = "predicted",
        msg = f"Result {label} with {score:.2f} of accuracy",
        prediction_id = prediction_id,
        label = label,
        score = score,
        time = 0
    )

class ModelFeedbackData(BaseModel):
    prediction_id: str
    label: str

    @field_validator('prediction_id')
    @classmethod
    def prediction_id_validator(cls, prediction_id: str) -> str:
        return prediction_id

    @field_validator('label')
    @classmethod
    def label_validator(cls, label: str) -> str:
        if not analyzer.validate_label(label):
            valid_labels = analyzer.get_valid_labels()
            raise ValueError(f"Label must be {", ".jsoin(valid_labels)}")
        return label.lower()
    
    
class ModelFeedbackResponse(BaseResponse):
    prediction_id: str
    label: str

@model_router.post("/feedback")
def model_feedback(data: ModelFeedbackData):
    # Todo:
    ## check prediction_id

    return ModelFeedbackResponse(
        status = "feedback accepted",
        message = "Thank you for you feedback",
        prediction_id = data.prediction_id,
        label = data.label
    )


@model_router.post("/train")
def model_train():
    return BaseResponse(
        status = "endpoint not available"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)