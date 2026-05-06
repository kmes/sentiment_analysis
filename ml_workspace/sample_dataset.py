from pathlib import Path

import pandas as pd
import typer

from common import ID2LABEL, LABELS, get_env, resolve_path

app = typer.Typer()

_DEFAULT_FROM = "data/simulation/train-00000-of-00001.parquet"
_DEFAULT_TO = "data/simulation/sample.parquet"


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna 'label' (str) al DataFrame qualunque sia il formato originale."""
    if "label_text" in df.columns:
        df = df.copy()
        df["label"] = df["label_text"].str.strip().str.lower()
    elif "label" in df.columns:
        if pd.api.types.is_integer_dtype(df["label"]):
            df = df.copy()
            df["label"] = df["label"].map(ID2LABEL)
        else:
            df = df.copy()
            df["label"] = df["label"].astype(str).str.strip().str.lower()
    else:
        raise ValueError(
            f"Colonna label non trovata. Colonne disponibili: {list(df.columns)}"
        )
    return df


@app.command()
def sample_dataset(
    from_dataset: Path = typer.Option(
        Path(_DEFAULT_FROM), "--from-dataset", help="Parquet sorgente"
    ),
    to_dataset: Path = typer.Option(
        Path(_DEFAULT_TO), "--to-dataset", help="Parquet di output"
    ),
    records: int = typer.Option(200, "--records", help="Numero di osservazioni da estrarre"),
    seed: int = typer.Option(int(get_env("SEED", "42")), "--seed", help="Seed per la riproducibilità"),
):
    """Campiona osservazioni con classi bilanciate da un parquet e produce un dataset per simulate_traffic.py."""
    from_path = resolve_path(from_dataset)
    to_path = resolve_path(to_dataset)

    if not from_path.exists():
        typer.echo(f"File non trovato: {from_path}", err=True)
        raise typer.Exit(1)

    try:
        df = pd.read_parquet(from_path)
        df = _normalize_labels(df)
    except Exception as e:
        typer.echo(f"Errore lettura dataset: {e}", err=True)
        raise typer.Exit(1)

    if "text" not in df.columns:
        typer.echo(f"Colonna 'text' non trovata. Colonne disponibili: {list(df.columns)}", err=True)
        raise typer.Exit(1)

    invalid = set(df["label"].dropna().unique()) - set(LABELS)
    if invalid:
        typer.echo(f"Label non riconosciute: {invalid}", err=True)
        raise typer.Exit(1)

    n_classes = len(LABELS)
    total_available = len(df)

    if records >= total_available:
        sample = df[["text", "label"]].copy()
        typer.echo(f"Richiesti {records} record, disponibili {total_available}: estrazione totale.")
    else:
        n_per_class = records // n_classes
        parts = [
            grp.sample(n=min(n_per_class, len(grp)), random_state=seed)
            for _, grp in df.groupby("label")
        ]
        sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        sample = sample[["text", "label"]]

    to_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(to_path, index=False)

    counts = sample["label"].value_counts()
    typer.echo(f"Dataset salvato: {to_path}")
    typer.echo(f"  Totale: {len(sample)} record")
    for lbl in LABELS:
        typer.echo(f"  {lbl}: {counts.get(lbl, 0)}")


if __name__ == "__main__":
    app()
