import time
from pathlib import Path
from typing import Optional

import httpx
import typer

from common import ID2LABEL, LABEL2ID, get_env, load_jsonl_dataset, resolve_path

app = typer.Typer()


def _load_parquet_dataset(path: Path) -> list[dict]:
    import pandas as pd

    df = pd.read_parquet(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Il Parquet deve avere colonne 'text' e 'label'. Trovate: {list(df.columns)}"
        )
    records = []
    for _, row in df.iterrows():
        text = str(row["text"])
        lbl = row["label"]
        if isinstance(lbl, str):
            key = lbl.strip().lower()
            if key not in LABEL2ID:
                continue
            label = key
        else:
            lid = int(lbl)
            if lid not in ID2LABEL:
                continue
            label = ID2LABEL[lid]
        records.append({"text": text, "label": label})
    return records


def _ensure_model_loaded(api_url: str) -> None:
    resp = httpx.get(f"{api_url}/status", timeout=10)
    resp.raise_for_status()
    if resp.json()["model"]["loaded"]:
        typer.echo("Modello già caricato.")
        return
    typer.echo("Modello non caricato. Caricamento in corso...")
    httpx.get(f"{api_url}/load-model", timeout=60).raise_for_status()
    resp2 = httpx.get(f"{api_url}/status", timeout=10)
    resp2.raise_for_status()
    if not resp2.json()["model"]["loaded"]:
        typer.echo("Errore: impossibile caricare il modello.", err=True)
        raise typer.Exit(1)
    typer.echo("Modello caricato.")


def _wait_for_prediction(api_url: str, prediction_id: str, max_attempts: int = 10) -> bool:
    for attempt in range(1, max_attempts + 1):
        resp = httpx.get(f"{api_url}/logs/predictions/{prediction_id}", timeout=10)
        if resp.status_code == 200:
            return True
        time.sleep(attempt * 0.01)  # 10 ms, 20 ms, ..., 100 ms
    return False


@app.command()
def simulate(
    dataset_path: Path = typer.Argument(
        Path("data/simulation/sample.parquet"),
        help="Path al dataset JSONL, directory JSONL o Parquet",
    ),
    api_url: str = typer.Option(
        get_env("SENTIMENT_API_URL", "http://localhost:8000"),
        "--api-url",
        help="URL base dell'API",
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="Numero massimo di record da processare"),
    delay: float = typer.Option(
        0.0, "--delay", help="Delay in secondi tra un record e il successivo"
    ),
):
    """Emula traffico predict+feedback sull'API a partire da un dataset etichettato."""
    dataset_path = resolve_path(dataset_path)

    if dataset_path.suffix.lower() == ".parquet":
        try:
            records = _load_parquet_dataset(dataset_path)
        except Exception as e:
            typer.echo(f"Errore caricamento Parquet: {e}", err=True)
            raise typer.Exit(1)
    else:
        try:
            records, _ = load_jsonl_dataset(dataset_path)
        except Exception as e:
            typer.echo(f"Errore caricamento dataset: {e}", err=True)
            raise typer.Exit(1)

    if not records:
        typer.echo("Nessun record valido nel dataset.", err=True)
        raise typer.Exit(1)

    if limit is not None:
        records = records[:limit]

    typer.echo(f"Dataset caricato: {len(records)} record")

    try:
        _ensure_model_loaded(api_url)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Errore verifica modello: {e}", err=True)
        raise typer.Exit(1)

    total = len(records)
    for i, record in enumerate(records, 1):
        text: str = record["text"]
        label: str = record["label"]

        try:
            pred_resp = httpx.post(f"{api_url}/model/predict", json={"text": text}, timeout=30)
            pred_resp.raise_for_status()
        except httpx.HTTPError as e:
            typer.echo(f"[{i}/{total}] ERRORE predict: {e}", err=True)
            continue

        pred_data = pred_resp.json()
        prediction_id = pred_data["prediction_id"]
        predicted_label = pred_data["predicted_label"]
        confidence = pred_data["confidence"]

        if not _wait_for_prediction(api_url, prediction_id):
            typer.echo(
                f"[{i}/{total}] ERRORE: prediction {prediction_id} non trovata in DB dopo 10 tentativi, salto.",
                err=True,
            )
            continue

        try:
            fb_resp = httpx.post(
                f"{api_url}/model/predictions/{prediction_id}/feedback",
                json={"label": label},
                timeout=30,
            )
            fb_resp.raise_for_status()
        except httpx.HTTPError as e:
            typer.echo(f"[{i}/{total}] ERRORE feedback: {e}", err=True)
            continue

        typer.echo(
            f'[{i}/{total}] text="{text[:50]}" → {predicted_label} ({confidence:.2f}) — feedback: {label}'
        )

        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    app()
