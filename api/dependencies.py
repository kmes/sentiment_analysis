from helpers.SentimentAnalyzer import SentimentAnalyzer
from helpers.TimerHelper import TimerHelper

import os

# Costanti lette da file .env
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME")
HF_MODEL_SAVE_PATH = os.getenv("HF_MODEL_SAVE_PATH")
HF_MODEL_VERSION = os.getenv("HF_MODEL_VERSION")

analyzer = SentimentAnalyzer(
    path=HF_MODEL_SAVE_PATH, 
    name=HF_MODEL_NAME, 
    version=HF_MODEL_VERSION
)
timer = TimerHelper()