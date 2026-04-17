from fastapi import HTTPException, status

from dependencies import analyzer

import os

def require_dev_env():
    app_env = os.environ.get("APP_ENV", "prod")
    if app_env.lower() != "dev":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is available only in the development environment."
        )

def check_model_loaded():
    if not analyzer.model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please load the model before making requests."
        )