from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_path):
        self.model = model_path
        self.tokenizer = model_path
        self.classifier = None

    def load_model(self):
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def analyze(self, text):
        result = self.classifier(text)[0]
        label = result['label']
        score = round(result['score'], 4)

        return label, score