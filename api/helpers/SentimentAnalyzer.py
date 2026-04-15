from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, model_path: str):
        self.model = model_path
        self.tokenizer = model_path
        self.classifier = None

    def load_model(self):
        self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def model_loaded(self) -> bool:
        return self.classifier is not None

    def unload_model(self):
        self.classifier = None

    def predict(self, text: str) -> tuple[str, float]:
        result = self.classifier(text)[0]
        label = result['label']
        score = round(result['score'], 4)

        return label, score
    
    def get_valid_labels(self) -> list[str]:
        return ["negative", "neutral", "positive"]
    
    def validate_label(self, label: str) -> bool:
        return label in self.get_valid_labels()