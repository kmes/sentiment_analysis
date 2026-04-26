from fastapi import APIRouter, BackgroundTasks, Depends, Query

from dependencies import analyzer, timer, model_load_time_ms, model_loaded
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

        background_tasks.add_task(
            services.save_model_load_log_background,
            model_load_id = analyzer.get_model_load_id(),
            model_name = analyzer.get_model_name(),
            model_version = analyzer.get_model_version(),
            load_time_ms = load_time
        )

        model_load_time_ms.observe(load_time)
        model_loaded.set(1)

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

        model_loaded.set(0)

        return LoadModelResponse(
            status = "model " + ("" if analyzer.model_loaded() else "not ") + "loaded",
            model = analyzer.get_model_info(),
            time = unload_time
        )