from fastapi import APIRouter

from dependencies import analyzer, timer
from schemas import BaseResponse, LoadModelResponse

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
            time = load_time
        )

@router.get("/unload-model")
def unload_model():
    if not analyzer.model_loaded():
        return BaseResponse(
            status = "already unloaded"
        )
    else:
        timer.reset_timer()
        analyzer.unload_model()
        unload_time = timer.partial_timer()

        return LoadModelResponse(
            status = "model " + ("" if analyzer.model_loaded() else "not ") + "loaded",
            time = unload_time
        )