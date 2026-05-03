from fastapi import APIRouter

from dependencies import feedback_disagreement_rate
from db import services

router = APIRouter(prefix="/internal")


@router.get("/refresh-metrics")
async def refresh_metrics():
    rate = await services.get_feedback_disagreement_rate(hours=24)
    feedback_disagreement_rate.set(rate)
    return {"status": "ok", "disagreement_rate": rate}
