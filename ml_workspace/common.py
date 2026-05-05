import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

WORKSPACE_DIR = Path(__file__).parent
load_dotenv(WORKSPACE_DIR / ".env")

LABELS = ("negative", "neutral", "positive")
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    return value


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE_DIR / p
    return p


@dataclass
class DatasetLoadReport:
    files: list[str] = field(default_factory=list)
    read: int = 0
    valid: int = 0
    skipped: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)

    def add_skip(self, reason: str) -> None:
        self.skipped += 1
        self.skipped_reasons[reason] = self.skipped_reasons.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "read": self.read,
            "valid": self.valid,
            "skipped": self.skipped,
            "skipped_reasons": self.skipped_reasons,
        }


def find_jsonl_files(path: Path) -> list[Path]:
    """Restituisce la lista di file JSONL: se path è file lo restituisce singolo,
    se è directory restituisce tutti i .jsonl ordinati alfabeticamente."""
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path non trovato: {path}")
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Nessun file .jsonl in {path}")
    return files


def parse_record(raw: dict) -> Optional[tuple[str, str]]:
    """Estrae (text, label) da un record JSONL. Restituisce None se non valido.
    Ignora campi extra (es. metadati con prefisso `_` da export_feedback)."""
    text = raw.get("text")
    label = raw.get("label")
    if not isinstance(text, str) or not text.strip():
        return None
    if not isinstance(label, str):
        return None
    label_norm = label.strip().lower()
    if label_norm not in LABEL2ID:
        return None
    return text, label_norm


def load_jsonl_dataset(path: Path) -> tuple[list[dict], DatasetLoadReport]:
    """Carica record da file/directory JSONL. Ogni record valido diventa
    {"text": str, "label": str, "label_id": int}."""
    files = find_jsonl_files(path)
    report = DatasetLoadReport(files=[str(f) for f in files])
    records: list[dict] = []

    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                report.read += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    report.add_skip("invalid_json")
                    continue
                parsed = parse_record(raw)
                if parsed is None:
                    report.add_skip("invalid_text_or_label")
                    continue
                text, label = parsed
                records.append(
                    {"text": text, "label": label, "label_id": LABEL2ID[label]}
                )
                report.valid += 1

    return records, report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def build_metrics_payload(
    *,
    model_version: str,
    eval_dataset: str,
    accuracy: float,
    f1_macro: float,
    f1_per_label: dict[str, float],
    eval_loss: float,
    num_samples: int,
) -> dict:
    """Costruisce il payload per POST /model/metrics (schema ModelMetricsPayload).
    f1_per_label deve contenere chiavi 'negative', 'neutral', 'positive'."""
    missing = [lbl for lbl in LABELS if lbl not in f1_per_label]
    if missing:
        raise ValueError(f"f1_per_label manca delle chiavi: {missing}")
    return {
        "model_version": model_version,
        "eval_dataset": eval_dataset,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_negative": float(f1_per_label["negative"]),
        "f1_neutral": float(f1_per_label["neutral"]),
        "f1_positive": float(f1_per_label["positive"]),
        "eval_loss": float(eval_loss),
        "num_samples": int(num_samples),
    }
