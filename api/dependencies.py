from helpers.SentimentAnalyzer import SentimentAnalyzer
from helpers.TimerHelper import TimerHelper

from prometheus_client import Histogram, Counter, Gauge

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

prediction_latency_ms = Histogram(
    "prediction_latency_ms",
    "Latency of prediction requests in milliseconds",
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

model_load_time_ms = Histogram(
    "model_load_time_ms",
    "Time taken to load the model in milliseconds",
    buckets=[500, 1000, 2000, 5000, 10000]
)

predictions_total = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["predicted_label"]
)

model_loaded = Gauge(
    "model_loaded",
    "Indicates if the model is currently loaded (1 = yes, 0 = no)"
)