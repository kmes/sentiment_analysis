from fastapi import APIRouter, BackgroundTasks, Depends, Query

from dependencies import analyzer, timer
from schemas import BaseResponse, LoadModelResponse
from db import services

import uuid

router = APIRouter()

@router.get("/load-model")
def load_model(background_tasks: BackgroundTasks):
    if analyzer.model_loaded():
        return BaseResponse(
            status = "already loaded"
        )
    else:
        model_load_id = uuid.uuid4()
        
        timer.reset_timer()
        analyzer.load_model(model_load_id)
        load_time = timer.partial_timer()

        model_name = "sentiment_analysis_model"
        background_tasks.add_task(
            services.save_model_load_log_background,
            model_load_id=model_load_id,
            model_name=model_name,
            model_version="v1.0",
            load_time_ms=load_time
        )

        return LoadModelResponse(
            status = "model " + ("" if analyzer.model_loaded() else "not ") + "loaded",
            model = analyzer.get_model_info(),
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
            model = analyzer.get_model_info(),
            time = unload_time
        )