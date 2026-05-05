import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from common import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    get_env,
    load_jsonl_dataset,
    resolve_path,
    write_json,
)

app = typer.Typer()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _split_train_val(records: list[dict], validation_split: float, seed: int):
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_val = int(len(records) * validation_split)
    val_idx = set(indices[:n_val])
    train = [records[i] for i in range(len(records)) if i not in val_idx]
    val = [records[i] for i in indices[:n_val]]
    return train, val


@app.command()
def fine_tune(
    candidate_version: str = typer.Argument(..., help="Versione del candidate model (es. v1.1.0-rc1)"),
    train_path: Optional[Path] = typer.Option(
        None, "--train-path", help="File JSONL o directory con gold dataset (default: GOLDEN_SET_DIR)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Directory di output dei runs (default: TRAIN_OUTPUT_DIR)"
    ),
    base_model: Optional[str] = typer.Option(
        None, "--base-model", help="Modello HF base (default: HF_BASE_MODEL)"
    ),
    max_length: int = typer.Option(
        int(get_env("MAX_LENGTH", "256")), "--max-length", help="Lunghezza massima tokenizzazione"
    ),
    batch_size: int = typer.Option(
        int(get_env("TRAIN_BATCH_SIZE", "16")), "--batch-size", help="Batch size training"
    ),
    epochs: int = typer.Option(
        int(get_env("NUM_EPOCHS", "10")), "--epochs", help="Numero epoche"
    ),
    learning_rate: float = typer.Option(
        float(get_env("LEARNING_RATE", "1e-3")), "--learning-rate", help="Learning rate"
    ),
    weight_decay: float = typer.Option(
        float(get_env("WEIGHT_DECAY", "0.01")), "--weight-decay", help="Weight decay"
    ),
    validation_split: float = typer.Option(
        float(get_env("VALIDATION_SPLIT", "0.2")), "--validation-split", help="Frazione val split interno"
    ),
    seed: int = typer.Option(int(get_env("SEED", "42")), "--seed", help="Seed riproducibilità"),
):
    """Classifier-only fine-tuning del modello base sul gold dataset cumulativo."""
    _set_seed(seed)

    train_path = train_path or Path(get_env("GOLDEN_SET_DIR", "datasets/fine-tuning"))
    output_root = resolve_path(output_dir or Path(get_env("TRAIN_OUTPUT_DIR", "runs")))
    base_model = base_model or get_env("HF_BASE_MODEL", "frasem/sentiment-analysis-roberta")

    typer.echo(f"Fine-tuning candidate {candidate_version}")
    typer.echo(f"  base model:       {base_model}")
    typer.echo(f"  train path:       {train_path}")
    typer.echo(f"  output dir:       {output_root}")

    records, load_report = load_jsonl_dataset(train_path)
    typer.echo(
        f"  dataset:          {load_report.read} letti, "
        f"{load_report.valid} validi, {load_report.skipped} scartati"
    )
    if load_report.skipped:
        typer.echo(f"  skip reasons:     {load_report.skipped_reasons}")
    if not records:
        typer.echo("Nessun record valido nel dataset.", err=True)
        raise typer.Exit(1)

    train_records, val_records = _split_train_val(records, validation_split, seed)
    typer.echo(f"  train/val split:  {len(train_records)} / {len(val_records)}")

    # Import pesanti dopo la validazione del dataset per fail-fast su errori CLI
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=False,
    )

    # Freeze backbone: lascia allenabile solo la classification head
    backbone = getattr(model, "roberta", None)
    if backbone is None:
        raise RuntimeError(
            "Modello senza attributo 'roberta': il freeze classifier-only "
            "richiede un backbone RoBERTa."
        )
    for param in backbone.parameters():
        param.requires_grad = False

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    typer.echo(f"  trainable params: {len(trainable)} tensori (classifier head)")

    def to_hf(records: list[dict]) -> Dataset:
        return Dataset.from_dict(
            {
                "text": [r["text"] for r in records],
                "label": [r["label_id"] for r in records],
            }
        )

    def tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=max_length
        )

    train_ds = to_hf(train_records).map(tokenize, batched=True, remove_columns=["text"])
    val_ds = to_hf(val_records).map(tokenize, batched=True, remove_columns=["text"]) if val_records else None

    run_dir = output_root / candidate_version
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"

    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="no",
        logging_strategy="epoch",
        seed=seed,
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if val_ds is not None else None,
    )

    train_result = trainer.train()
    typer.echo(f"Training completato. loss finale: {train_result.training_loss:.4f}")

    eval_metrics: dict = {}
    if val_ds is not None:
        eval_metrics = trainer.evaluate()
        typer.echo(f"Eval interna: {eval_metrics}")

    # Salva modello + tokenizer in run_dir
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    summary = {
        "candidate_version": candidate_version,
        "base_model": base_model,
        "dataset": load_report.to_dict(),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "hyperparameters": {
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "validation_split": validation_split,
            "seed": seed,
        },
        "training_loss": train_result.training_loss,
        "eval_metrics": eval_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "training_summary.json", summary)
    typer.echo(f"Artefatti salvati in: {run_dir}")
    typer.echo(f"Training summary: {run_dir / 'training_summary.json'}")


if __name__ == "__main__":
    app()
