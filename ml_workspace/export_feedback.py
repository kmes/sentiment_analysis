import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import typer
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

app = typer.Typer()

API_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
FEEDBACK_EXPORT_ENDPOINT = os.getenv("FEEDBACK_EXPORT_ENDPOINT", "/model/feedback-export")
BRONZE_SET_DIR = os.getenv("BRONZE_SET_DIR", "datasets/raw")

@app.command()
def export(
    model_version: str = typer.Argument(..., help="Versione del modello (es. v1.0.0)"),
    date_from: Optional[str] = typer.Option(None, "--date-from", help="Data inizio ISO 8601"),
    date_to: Optional[str] = typer.Option(None, "--date-to", help="Data fine ISO 8601"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Numero massimo di record"),
    output_dir: Path = typer.Option(
        Path(BRONZE_SET_DIR), "--output-dir", help="Directory di output"
    ),
):
    params = {"model_version": model_version}
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    if limit:
        params["limit"] = limit

    typer.echo(f"Esportazione feedback per {model_version} da {API_URL}...")

    try:
        response = httpx.get(f"{API_URL}{FEEDBACK_EXPORT_ENDPOINT}", params=params, timeout=30)
        response.raise_for_status()
    except httpx.HTTPError as e:
        typer.echo(f"Errore HTTP: {e}", err=True)
        raise typer.Exit(1)

    data = response.json()
    total = data["total_records"]
    typer.echo(f"Record ricevuti: {total}")

    if total == 0:
        typer.echo("Nessun record da esportare.")
        raise typer.Exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_version = model_version.replace("/", "-")

    jsonl_path = output_dir / f"feedback_{safe_version}_{ts}.jsonl"
    manifest_path = output_dir / f"manifest_{safe_version}_{ts}.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in data["records"]:
            row = {
                "text": record["input_text"],
                "label": record["true_label"],
                "_predicted_label": record["predicted_label"],
                "_confidence": record["confidence"],
                "_prediction_id": record["prediction_id"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "model_version": data["model_version"],
        "exported_at": data["exported_at"],
        "date_from_effective": data["date_from_effective"],
        "date_to_effective": data["date_to_effective"],
        "total_records": total,
        "limit_applied": data["limit_applied"],
        "output_file": str(jsonl_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    typer.echo(f"JSONL:     {jsonl_path}")
    typer.echo(f"Manifest:  {manifest_path}")


if __name__ == "__main__":
    app()
