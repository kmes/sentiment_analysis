from fastapi import FastAPI

from .helpers.SentimentAnalyzer import SentimentAnalyzer
from .helpers.TimerHelper import TimerHelper

from .routes import routes
from .db.services import lifespan

analyzer = SentimentAnalyzer("cardiffnlp/twitter-roberta-base-sentiment-latest")

timer = TimerHelper()

app = FastAPI(title="Sentiment API", lifespan=lifespan)
app.include_router(routes)