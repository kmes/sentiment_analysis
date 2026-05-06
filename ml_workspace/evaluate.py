from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import typer

from common import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    build_metrics_payload,
    get_env,
    resolve_path,
    write_json,
)

app = typer.Typer()


def _load_eval_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Carica eval set Parquet con colonne 'text' (str) e 'label' (int o str)."""
    import pandas as pd

    df = pd.read_parquet(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Eval set deve avere colonne 'text' e 'label'. Trovate: {list(df.columns)}"
        )

    texts = df["text"].astype(str).tolist()
    raw_labels = df["label"].tolist()

    labels: list[int] = []
    for lbl in raw_labels:
        if isinstance(lbl, str):
            key = lbl.strip().lower()
            if key not in LABEL2ID:
                raise ValueError(f"Label non valida nel Parquet: {lbl!r}")
            labels.append(LABEL2ID[key])
        else:
            lid = int(lbl)
            if lid not in ID2LABEL:
                raise ValueError(f"Label id fuori range: {lid}")
            labels.append(lid)

    return texts, labels


def _compute_metrics(
    *, predictions: list[int], labels: list[int], eval_loss: float, num_samples: int
) -> dict:
    """Calcola accuracy, f1_macro, f1 per label e ritorna dict pronto per build_metrics_payload."""
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = float(accuracy_score(labels, predictions))
    f1_macro = float(f1_score(labels, predictions, average="macro", zero_division=0))
    f1_per_label_arr = f1_score(
        labels,
        predictions,
        labels=[LABEL2ID[l] for l in LABELS],
        average=None,
        zero_division=0,
    )
    f1_per_label = {label: float(score) for label, score in zip(LABELS, f1_per_label_arr)}

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_per_label": f1_per_label,
        "eval_loss": float(eval_loss),
        "num_samples": int(num_samples),
    }


def post_metrics_to_api(api_url: str, payload: dict) -> dict:
    """Invia payload a {api_url}/model/metrics. Ritorna dict con esito.
    Non solleva eccezioni: in caso di errore HTTP ritorna {'sent': False, 'error': ...}."""
    url = f"{api_url}/model/metrics"
    try:
        response = httpx.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as e:
        return {"sent": False, "url": url, "error": str(e)}
    return {"sent": True, "url": url, "status_code": response.status_code}


def _run_inference(
    model_dir: Path, texts: list[int], labels: list[int], batch_size: int, max_length: int
):
    """Esegue inference su tutti i testi e calcola eval_loss + predizioni."""
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_preds: list[int] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_labels = labels[start : start + batch_size]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**enc).logits
            label_tensor = torch.tensor(batch_labels, device=device, dtype=torch.long)
            total_loss += loss_fn(logits, label_tensor).item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(int(x) for x in preds)

    eval_loss = total_loss / len(texts) if texts else 0.0
    return all_preds, eval_loss


@app.command()
def evaluate(
    candidate_version: str = typer.Argument(..., help="Versione del candidate model (es. v1.1.0-rc1)"),
    model_dir: Optional[Path] = typer.Option(
        None, "--model-dir", help="Directory del candidate (default: <TRAIN_OUTPUT_DIR>/<version>)"
    ),
    eval_set: Optional[Path] = typer.Option(
        None, "--eval-set", help="Path eval set Parquet (default: EVAL_SET_PATH)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory report (default: EVAL_RESULTS_DIR)"
    ),
    batch_size: int = typer.Option(
        int(get_env("EVAL_BATCH_SIZE", "32")), "--batch-size", help="Batch size inference"
    ),
    max_length: int = typer.Option(
        int(get_env("MAX_LENGTH", "256")), "--max-length", help="Lunghezza massima tokenizzazione"
    ),
    report_only: bool = typer.Option(
        False, "--report-only", help="Non inviare le metriche all'API"
    ),
):
    """Valuta un candidate model sull'eval set Parquet e (opzionalmente) invia le metriche all'API."""
    train_output_dir = Path(get_env("TRAIN_OUTPUT_DIR", "runs"))
    model_dir = resolve_path(model_dir or train_output_dir / candidate_version)
    eval_set = resolve_path(eval_set or Path(get_env("EVAL_SET_PATH", "data/evaluate/eval.parquet")))
    output_dir = resolve_path(output_dir or Path(get_env("EVAL_RESULTS_DIR", "artifacts/evaluations")))

    if not model_dir.exists():
        typer.echo(f"Model dir non trovata: {model_dir}", err=True)
        raise typer.Exit(1)
    if not eval_set.exists():
        typer.echo(f"Eval set non trovato: {eval_set}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Evaluation candidate {candidate_version}")
    typer.echo(f"  model dir:        {model_dir}")
    typer.echo(f"  eval set:         {eval_set}")
    typer.echo(f"  output dir:       {output_dir}")

    texts, labels = _load_eval_dataset(eval_set)
    typer.echo(f"  num samples:      {len(texts)}")

    predictions, eval_loss = _run_inference(
        model_dir=model_dir,
        texts=texts,
        labels=labels,
        batch_size=batch_size,
        max_length=max_length,
    )

    metrics = _compute_metrics(
        predictions=predictions,
        labels=labels,
        eval_loss=eval_loss,
        num_samples=len(texts),
    )
    typer.echo(
        f"  accuracy: {metrics['accuracy']:.4f}  f1_macro: {metrics['f1_macro']:.4f}  "
        f"eval_loss: {metrics['eval_loss']:.4f}"
    )
    typer.echo(f"  f1 per label:     {metrics['f1_per_label']}")

    payload = build_metrics_payload(
        model_version=candidate_version,
        eval_dataset=str(eval_set.name),
        accuracy=metrics["accuracy"],
        f1_macro=metrics["f1_macro"],
        f1_per_label=metrics["f1_per_label"],
        eval_loss=metrics["eval_loss"],
        num_samples=metrics["num_samples"],
    )

    api_result: dict = {"sent": False}
    if not report_only:
        api_url = get_env("SENTIMENT_API_URL", "http://localhost:8000")
        api_result = post_metrics_to_api(api_url, payload)
        if api_result["sent"]:
            typer.echo(f"Metriche inviate a {api_result['url']}")
        else:
            typer.echo(f"Errore invio metriche: {api_result.get('error')}", err=True)

    report = {
        "candidate_version": candidate_version,
        "model_dir": str(model_dir),
        "eval_dataset": str(eval_set),
        "metrics": payload,
        "api_post": api_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / f"{candidate_version}.json"
    write_json(report_path, report)
    typer.echo(f"Report locale: {report_path}")

    if not report_only and not api_result.get("sent"):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
