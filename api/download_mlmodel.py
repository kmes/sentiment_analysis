import os
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
save_path = "./ml_models/twitter-roberta-sentiment"

print(f"[1/4] Avvio download di: {model_name}", flush=True)

try:
    os.makedirs(save_path, exist_ok=True)
    print(f"[2/4] Cartella creata: {save_path}", flush=True)

    print("[3/4] Download tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)
    print("      Tokenizer salvato.", flush=True)

    print("[3/4] Download modello (può richiedere qualche minuto)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(save_path)
    print("      Modello salvato.", flush=True)

    print(f"\n[4/4] ✅ Completato! File in: {os.path.abspath(save_path)}", flush=True)
    print("File salvati:")
    for f in os.listdir(save_path):
        print(f"  - {f}")

except Exception as e:
    print(f"\n❌ Errore: {e}", file=sys.stderr, flush=True)
    sys.exit(1)