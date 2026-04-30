import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = os.getenv("HF_MODEL_NAME")
save_path = os.getenv("HF_MODEL_SAVE_PATH")
revision = os.getenv("HF_MODEL_REVISION") or "main"

revision_lock_path = os.path.join(save_path, ".hf_revision")

print(f"[1/4] Avvio download di: {model_name} @ {revision}", flush=True)

if os.path.exists(os.path.join(save_path, "config.json")):
    saved_revision = None
    if os.path.exists(revision_lock_path):
        with open(revision_lock_path) as f:
            saved_revision = f.read().strip()
    if saved_revision == revision:
        print(f"Modello già presente in {save_path} @ {revision}. Download saltato.", flush=True)
        sys.exit(0)
    else:
        print(f"Revisione cambiata ({saved_revision} → {revision}). Re-download.", flush=True)

try:
    os.makedirs(save_path, exist_ok=True)
    print(f"[2/4] Cartella creata: {save_path}", flush=True)

    print("[3/4] Download tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.save_pretrained(save_path)
    print("      Tokenizer salvato.", flush=True)

    print("[3/4] Download modello (può richiedere qualche minuto)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    model.save_pretrained(save_path)
    print("      Modello salvato.", flush=True)

    with open(revision_lock_path, "w") as f:
        f.write(revision)

    print(f"\n[4/4] ✅ Completato! File in: {os.path.abspath(save_path)}", flush=True)
    print("File salvati:")
    for f in os.listdir(save_path):
        print(f"  - {f}")

except Exception as e:
    print(f"\n❌ Errore: {e}", file=sys.stderr, flush=True)
    sys.exit(1)