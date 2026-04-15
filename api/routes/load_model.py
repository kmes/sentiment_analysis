from fastapi import APIRouter

from ..boostrap import analyzer, timer
from ..schemas import BaseResponse, LoadModelResponse

router = APIRouter()

@router.get("/load-model")
def load_model():
    if analyzer.model_loaded():
        return BaseResponse(
            status = "already loaded"
        )
    else:
        timer.reset_timer()
        analyzer.load_model()
        load_time = timer.partial_timer()

        return LoadModelResponse(
            status = "model " + ("" if analyzer.model_loaded() else "not ") + "loaded",
            loaded_time = load_time
        )