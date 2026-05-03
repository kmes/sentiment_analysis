from fastapi import APIRouter

from .status import router as status_router
from .load_model import router as load_model_router
from .model import router as model_router, data_router as model_data_router
from .logs import router as logs_router
from .internal import router as internal_router

routes = APIRouter()

routes.include_router(status_router, tags=["Status"])
routes.include_router(load_model_router, tags=["Load Model"])
routes.include_router(model_router, tags=["Model"])
routes.include_router(model_data_router, tags=["Model"])
routes.include_router(logs_router, tags=["Logs"])
routes.include_router(internal_router, tags=["Internal"])