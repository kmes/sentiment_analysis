from fastapi import APIRouter

from dependencies import analyzer, timer

from schemas import StatusResponse

router = APIRouter()

@router.get("/status")
def status_endopoint():
    return StatusResponse(
        status = "ok",
        model = ("" if analyzer.model_loaded() else "not ") + "loaded",
        uptime = timer.get_uptime()
    )