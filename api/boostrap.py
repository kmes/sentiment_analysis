from fastapi import FastAPI

from routes import routes
from db.services import lifespan

from dependencies import timer

app = FastAPI(title="Sentiment API", lifespan=lifespan)
app.include_router(routes)
timer.init()