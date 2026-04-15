from fastapi import APIRouter

from .status import router as status_router
from .load_model import router as load_model_router
from .model import router as model_router

routes = APIRouter()

routes.include_router(status_router, tags=["Status"])
routes.include_router(load_model_router, tags=["Load Model"])
routes.include_router(model_router, tags=["Model"])