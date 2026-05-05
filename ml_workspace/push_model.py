from pathlib import Path
from typing import Optional

import typer

from common import get_env, resolve_path

app = typer.Typer()


@app.command()
def push_model(
    version: str = typer.Argument(..., help="Versione del modello da pubblicare (es. v1.1.0)"),
    model_dir: Optional[Path] = typer.Option(
        None, "--model-dir", help="Directory del candidate (default: <TRAIN_OUTPUT_DIR>/<version>)"
    ),
    repo_id: Optional[str] = typer.Option(
        None, "--repo-id", help="Repo HF di destinazione (default: HF_REPO_ID)"
    ),
    create_tag: bool = typer.Option(
        False, "--create-tag", help="Crea il tag <version> nel repo HF dopo il push"
    ),
    commit_message: Optional[str] = typer.Option(
        None, "--commit-message", help="Commit message HF (default: 'Add fine-tuned candidate <version>')"
    ),
):
    """Pubblica un candidate model approvato nel repo HuggingFace e crea opzionalmente un tag di versione."""
    hf_token = get_env("HF_TOKEN")
    if not hf_token:
        typer.echo(
            "HF_TOKEN non configurato. Impostarlo in ml_workspace/.env o come variabile d'ambiente.",
            err=True,
        )
        raise typer.Exit(1)

    repo_id = repo_id or get_env("HF_REPO_ID", "frasem/sentiment-analysis-roberta")
    train_output_dir = Path(get_env("TRAIN_OUTPUT_DIR", "runs"))
    model_dir = resolve_path(model_dir or train_output_dir / version)
    commit_message = commit_message or f"Add fine-tuned candidate {version}"

    if not model_dir.exists():
        typer.echo(f"Model dir non trovata: {model_dir}", err=True)
        raise typer.Exit(1)
    if not (model_dir / "config.json").exists():
        typer.echo(
            f"config.json non trovato in {model_dir}. "
            "Verificare che fine_tune.py sia stato eseguito correttamente.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Push candidate {version}")
    typer.echo(f"  model dir:  {model_dir}")
    typer.echo(f"  repo:       {repo_id}")
    typer.echo(f"  create tag: {create_tag}")

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    typer.echo("Caricamento file sul repo HF...")
    commit_info = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    typer.echo(f"Push completato: {commit_info.commit_url}")

    if create_tag:
        api.create_tag(
            repo_id=repo_id,
            tag=version,
            repo_type="model",
        )
        tag_url = f"https://huggingface.co/{repo_id}/tree/{version}"
        typer.echo(f"Tag creato:      {tag_url}")

    typer.echo(f"\nModello {version} pubblicato su {repo_id}.")
    if create_tag:
        typer.echo(
            f"\nPasso successivo: aggiornare HF_MODEL_REVISION={version} nel .env del progetto "
            "e riavviare l'API per attivare il re-download."
        )


if __name__ == "__main__":
    app()
