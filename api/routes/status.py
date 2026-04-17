from fastapi import APIRouter

from dependencies import analyzer, timer

from schemas import StatusResponse

router = APIRouter()

@router.get("/status")
def status_endopoint():
    return StatusResponse(
        status = "ok",
        model = analyzer.get_model_info(),
        uptime = timer.get_uptime()
    )