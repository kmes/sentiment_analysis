from fastapi import FastAPI
from prometheus_client import make_asgi_app

from routes import routes
from db.services import lifespan

from dependencies import timer

app = FastAPI(title="Sentiment API", lifespan=lifespan)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
app.include_router(routes)
timer.init()