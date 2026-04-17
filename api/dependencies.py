from helpers.SentimentAnalyzer import SentimentAnalyzer
from helpers.TimerHelper import TimerHelper

analyzer = SentimentAnalyzer(
    path="./ml_models/twitter-roberta-sentiment", 
    name="twitter-roberta-sentiment", 
    version="1.0.0"
)
timer = TimerHelper()