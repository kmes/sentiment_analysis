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

prediction_text_length = Histogram(
    "prediction_text_length_chars",
    "Lunghezza in caratteri del testo di input alle predizioni",
    buckets=[10, 25, 50, 100, 200, 500, 1000, 2000]
)

prediction_confidence = Histogram(
    "prediction_confidence",
    "Confidence score delle predizioni",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

feedback_disagreement_rate = Gauge(
    "feedback_disagreement_rate",
    "Percentuale feedback con true_label != predicted_label (finestra rolling)"
)

feedback_total = Counter(
    "feedback_total",
    "Contatore feedback ricevuti per true_label",
    ["true_label"]
)

model_accuracy = Gauge("model_accuracy", "Accuracy ultima valutazione", ["model_version"])
model_f1_macro = Gauge("model_f1_macro", "F1 macro ultima valutazione", ["model_version"])
model_f1_per_label = Gauge(
    "model_f1_per_label", "F1 per label ultima valutazione", ["label", "model_version"]
)