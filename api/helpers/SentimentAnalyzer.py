from transformers import pipeline

import uuid

class SentimentAnalyzer:
    def __init__(self, path: str, name: str, version: str):
        self._model_name = name
        self._model_version = version
        self._model_path = path
        self._classifier = None
        self._model_load_id = None

    def get_model_load_id(self) -> uuid.UUID | None:
        return self._model_load_id

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_version(self) -> str:
        return self._model_version

    def get_model_info(self) -> dict:
        return {
            "loaded": self.model_loaded(),
            "load_id": self.get_model_load_id(),
            "name": self.get_model_name(),
            "version": self.get_model_version()
        }

    def load_model(self, model_load_id: uuid.UUID):
        self._classifier = pipeline(
            "sentiment-analysis",
            model=self._model_path,
            tokenizer=self._model_path,
            local_files_only=True,   # forza l'uso solo della cache locale
        )
        self._model_load_id = model_load_id

    def model_loaded(self) -> bool:
        return self._classifier is not None

    def unload_model(self):
        self._classifier = None
        self._model_load_id = None

    def predict(self, text: str) -> tuple[str, float]:
        result = self._classifier(text)[0]
        label = result["label"].lower()  # normalizza a minuscolo
        score = round(result["score"], 4)
        return label, score

    def get_valid_labels(self) -> list[str]:
        return ["negative", "neutral", "positive"]

    def validate_label(self, label: str) -> bool:
        return label in self.get_valid_labels()