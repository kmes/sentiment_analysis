# ml_workspace — Guida operativa

Area di lavoro per il fine-tuning manuale del modello `frasem/sentiment-analysis-roberta`. Contiene gli script CLI per esportare i feedback, addestrare un candidate model, valutarlo e pubblicarlo su Hugging Face.

---

## Indice

1. [Struttura della cartella](#struttura-della-cartella)
2. [Setup iniziale](#setup-iniziale)
3. [Eseguire i test](#eseguire-i-test)
4. [Flusso di fine-tuning end-to-end](#flusso-di-fine-tuning-end-to-end)
   - [Step 1 — Esporta i feedback](#step-1--esporta-i-feedback)
   - [Step 2 — Prepara il gold dataset](#step-2--prepara-il-gold-dataset)
   - [Step 3 — Fine-tuning](#step-3--fine-tuning)
   - [Step 4 — Valutazione](#step-4--valutazione)
   - [Step 5 — Revisione umana](#step-5--revisione-umana)
   - [Step 6 — Push su Hugging Face](#step-6--push-su-hugging-face)
   - [Step 7 — Deploy in produzione](#step-7--deploy-in-produzione)
5. [Riferimento variabili d'ambiente](#riferimento-variabili-dambiente)
6. [Riferimento CLI completo](#riferimento-cli-completo)

---

## Struttura della cartella

```
ml_workspace/
├── .env                    # Variabili d'ambiente (da configurare prima di iniziare)
├── requirements.txt        # Dipendenze Python del workspace
├── common.py               # Modulo condiviso (label mapping, parsing, utils)
├── export_feedback.py      # Step 1: esporta feedback dall'API
├── fine_tune.py            # Step 3: addestra il candidate model
├── evaluate.py             # Step 4: valuta il candidate model
├── push_model.py           # Step 6: pubblica il modello su Hugging Face
├── tests/                  # Test unitari
├── model/                  # [submodule] Clone del modello base frasem/sentiment-analysis-roberta
├── runs/                   # Candidate models prodotti da fine_tune.py; push_model.py pubblica da qui su HF (model/ è il modello sorgente, non viene mai toccato)
├── artifacts/
│   └── evaluations/        # Report JSON di valutazione (creato automaticamente)
└── data/
    ├── raw/                # Bronze: feedback esportati dall'API (JSONL + manifest)
    ├── silver/             # Silver: dati filtrati (cura del data scientist)
    ├── fine-tuning/        # Gold: dataset validato per il training
    ├── evaluate/           # Eval set Parquet usato da evaluate.py (data/evaluate/eval.parquet)
    ├── simulation/         # Dataset per simulate_traffic.py (sorgente + sample bilanciato)
    └── sentiment-dataset/  # [submodule] frasem/sentiment-dataset — sorgente Parquet train/test/validation
```

> **Nota sui submodule**: `model/` e `data/sentiment-dataset/` sono git submoduli. Dopo un clone del repo vanno inizializzati esplicitamente (vedi Setup iniziale).

---

## Setup iniziale

### 1. Creare il file `.env`

Copiare il template e compilare le variabili necessarie:

```bash
cp ml_workspace/.env_sample ml_workspace/.env
```

Aprire `.env` e impostare almeno `HF_TOKEN` (obbligatorio per lo Step 6 — Push su Hugging Face):

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Le altre variabili hanno valori di default funzionanti; modificarle solo se si usa un repo o una configurazione diversa da quella standard.

---

### 2. Inizializzare i submodule git

Dopo aver clonato il repo principale, i submoduli `model/` e `data/sentiment-dataset/` sono vuoti. Inizializzarli dalla root del progetto (`sentiment_analysis/`):

```bash
git submodule update --init --recursive
```

Se è il primo clone del progetto, si può includere direttamente:

```bash
git clone --recurse-submodules <repo-url>
```

### 3. Posizionarsi nella cartella

```bash
cd ml_workspace
```

### 4. Attivare il virtual environment (o crearlo se non esiste)

```bash
# Se il venv esiste già
source .venv/bin/activate

# Se non esiste, crearlo e installare le dipendenze
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. Configurare le variabili d'ambiente

Il file `.env` viene caricato automaticamente da tutti gli script. Verificare che le variabili principali siano corrette:

```bash
cat .env
```

Le variabili rilevanti per il flusso di fine-tuning:

| Variabile | Default | Quando serve |
|---|---|---|
| `SENTIMENT_API_URL` | `http://localhost:8000` | Sempre |
| `GOLDEN_SET_DIR` | `data/fine-tuning` | Step 3 |
| `EVAL_SET_PATH` | `data/evaluate/eval.parquet` | Step 4 |
| `HF_BASE_MODEL` | `frasem/sentiment-analysis-roberta` | Step 3 |
| `HF_REPO_ID` | `frasem/sentiment-analysis-roberta` | Step 6 |
| `HF_TOKEN` | _(vuoto)_ | Step 6 — impostare prima del push |

Per lo Step 6 è obbligatorio impostare `HF_TOKEN`:

```bash
# In .env, aggiungere o aggiornare:
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Eseguire i test

I test sono unitari puri: non richiedono l'API in esecuzione, connessione di rete, né il download di modelli ML. Si possono lanciare in qualsiasi momento.

```bash
python -m pytest tests/ -v
```

Output atteso (21 test):

```
tests/test_common.py::test_label_mapping_is_stable PASSED
tests/test_common.py::test_parse_record_valid[...] PASSED  (x4)
tests/test_common.py::test_parse_record_invalid[...] PASSED  (x7)
tests/test_common.py::test_load_jsonl_dataset_directory PASSED
tests/test_common.py::test_load_jsonl_dataset_single_file PASSED
tests/test_common.py::test_find_jsonl_files_missing PASSED
tests/test_common.py::test_find_jsonl_files_empty_dir PASSED
tests/test_common.py::test_build_metrics_payload_complete PASSED
tests/test_common.py::test_build_metrics_payload_missing_label_raises PASSED
tests/test_common.py::test_write_json_creates_parent_dirs PASSED
tests/test_common.py::test_post_metrics_to_api_handles_connection_error PASSED
tests/test_common.py::test_post_metrics_to_api_success PASSED
============================== 21 passed ============================
```

---

## Flusso di fine-tuning end-to-end

Il flusso parte dal presupposto che il modello `v1.0.0` sia già in produzione e che gli utenti abbiano fornito feedback sulle predizioni errate.

```
[API in produzione] → export_feedback → [gold dataset] → fine_tune → evaluate → push_model → [nuova versione in produzione]
```

---

### Step 1 — Esporta i feedback

Scarica dall'API i feedback raccolti dagli utenti per la versione del modello attualmente in produzione. L'API deve essere raggiungibile.

```bash
python export_feedback.py v1.0.0
```

Con filtri opzionali su date e quantità:

```bash
python export_feedback.py v1.0.0 \
  --date-from 2026-01-01T00:00:00 \
  --date-to 2026-05-01T00:00:00 \
  --limit 500
```

**Output** in `data/raw/`:

```
data/raw/
├── feedback_v1.0.0_20260503T120000Z.jsonl
└── manifest_v1.0.0_20260503T120000Z.json
```

Ogni riga del JSONL ha questa struttura (i campi `_` sono metadati di review, non entrano nel training):

```json
{"text": "Great product!", "label": "positive", "_predicted_label": "neutral", "_confidence": 0.61, "_prediction_id": "..."}
```

---

### Step 2 — Prepara il gold dataset

> Questo step è manuale e di responsabilità del data scientist.

Partendo dai file in `data/raw/`, rivedere i record e produrre il gold dataset validato in `data/fine-tuning/`. Il formato richiesto è JSONL con solo due campi:

```json
{"text": "Great product!", "label": "positive"}
{"text": "Terrible experience.", "label": "negative"}
```

Le label valide sono esattamente tre: `negative`, `neutral`, `positive`.

È possibile avere più file JSONL nella stessa cartella: `fine_tune.py` li legge tutti in modo cumulativo (gold dataset di sessioni precedenti incluso).

---

### Step 3 — Fine-tuning

Addestra la classification head del modello base sul gold dataset. Il backbone RoBERTa viene congelato: si aggiornano solo i pesi dell'ultimo layer di classificazione.

```bash
python fine_tune.py v1.1.0-rc1 \
  --train-path data/fine-tuning \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 1e-3 \
  --validation-split 0.2 \
  --seed 42
```

Lo script stampa un report di caricamento dataset (record letti / validi / scartati), il numero di parametri allenabili e le metriche di validation interna alla fine di ogni epoca.

**Output**: i pesi del modello fine-tunato vengono salvati in `runs/<candidate_version>/`. La cartella è creata automaticamente da `fine_tune.py`. Il clone locale in `model/` è il modello sorgente e non viene mai toccato o sovrascritto.

---

### Step 4 — Valutazione

Valuta il candidate model sull'eval set Parquet (`data/evaluate/eval.parquet`) e invia le metriche all'API (vengono salvate nel DB e aggiornano la dashboard Grafana).

```bash
python evaluate.py v1.1.0-rc1 \
  --model-dir runs/v1.1.0-rc1
```

Se l'API non è raggiungibile o si vuole solo il report locale senza inviare le metriche:

```bash
python evaluate.py v1.1.0-rc1 \
  --model-dir runs/v1.1.0-rc1 \
  --report-only
```

**Output** in `artifacts/evaluations/v1.1.0-rc1.json`:

```json
{
  "candidate_version": "v1.1.0-rc1",
  "model_dir": ".../runs/v1.1.0-rc1",
  "eval_dataset": ".../test-00000-of-00001.parquet",
  "metrics": {
    "accuracy": 0.87,
    "f1_macro": 0.85,
    "f1_negative": 0.83,
    "f1_neutral": 0.81,
    "f1_positive": 0.91,
    "eval_loss": 0.38,
    "num_samples": 12284
  },
  "api_post": {"sent": true, "status_code": 201},
  "timestamp": "2026-05-03T14:30:00+00:00"
}
```

> Il report viene scritto su disco anche se l'invio all'API fallisce.

---

### Step 5 — Revisione umana

Leggere il report di valutazione e confrontarlo con le metriche del modello in produzione:

```bash
cat artifacts/evaluations/v1.1.0-rc1.json
```

Se le metriche sono soddisfacenti si procede con lo Step 6. Altrimenti si torna allo Step 2 per ampliare o correggere il gold dataset e ripetere il training.

---

### Step 6 — Push su Hugging Face

Pubblica il candidate approvato sul repo HF e crea il tag di versione `v1.1.0`. Richiede `HF_TOKEN` configurato in `.env`.

```bash
python push_model.py v1.1.0 \
  --model-dir runs/v1.1.0-rc1 \
  --create-tag
```

**Cosa viene pushato e dove:**

- **Sorgente**: la cartella `runs/v1.1.0-rc1/` — i pesi prodotti da `fine_tune.py`, non il clone in `model/`. La cartella `model/` non viene mai toccata da questo script: serve solo come base per il training.
- **Destinazione**: il repo HuggingFace definito da `HF_REPO_ID` nel `.env` (default: `frasem/sentiment-analysis-roberta`).

Lo script esegue questi passi in sequenza:

1. Verifica che `HF_TOKEN` sia presente — senza token l'operazione fallisce subito.
2. Controlla che `runs/v1.1.0-rc1/config.json` esista (guard contro push di cartelle vuote o errate).
3. Carica l'intera cartella sul repo HF come singolo commit (`upload_folder`).
4. Con `--create-tag`: crea il tag `v1.1.0` nel repo HF — è questo tag che l'API userà per scaricare la versione corretta tramite `HF_MODEL_REVISION`.
5. Stampa l'URL del commit e del tag, e ricorda di aggiornare `HF_MODEL_REVISION` nel `.env` del progetto.

---

### Step 7 — Deploy in produzione

Aggiornare le variabili nel `.env` principale del progetto (nella root di `sentiment_analysis/`, non in `ml_workspace/`):

```bash
# sentiment_analysis/.env
HF_MODEL_REVISION=v1.1.0
HF_MODEL_VERSION=v1.1.0
```

Riavviare l'API. All'avvio, `download_mlmodel.py` rileva che la revision è cambiata e scarica automaticamente i nuovi pesi da HF:

```bash
docker compose up -d --build api
```

Caricare il modello in memoria:

```bash
curl http://localhost:8000/load-model
```

Il nuovo modello è in produzione.

---

## Riferimento variabili d'ambiente

Tutte le variabili sono in `.env` e vengono caricate automaticamente da ogni script.

| Variabile | Default | Descrizione |
|---|---|---|
| `SENTIMENT_API_URL` | `http://localhost:8000` | URL base dell'API |
| `FEEDBACK_EXPORT_ENDPOINT` | `/model/feedback-export` | Endpoint export feedback |
| `BRONZE_SET_DIR` | `data/raw` | Output di `export_feedback.py` |
| `SILVER_SET_DIR` | `data/silver` | Dataset silver (uso manuale) |
| `GOLDEN_SET_DIR` | `data/fine-tuning` | Gold dataset per il training |
| `EVAL_SET_PATH` | `data/evaluate/eval.parquet` | Eval set per `evaluate.py` |
| `MODEL_PATH` | `model` | Clone locale del modello base |
| `HF_BASE_MODEL` | `frasem/sentiment-analysis-roberta` | Modello di partenza per il fine-tuning |
| `HF_REPO_ID` | `frasem/sentiment-analysis-roberta` | Repo HF di destinazione per il push |
| `TRAIN_OUTPUT_DIR` | `runs` | Directory output di `fine_tune.py` |
| `EVAL_RESULTS_DIR` | `artifacts/evaluations` | Directory report di `evaluate.py` |
| `MAX_LENGTH` | `256` | Lunghezza massima tokenizzazione |
| `TRAIN_BATCH_SIZE` | `16` | Batch size training |
| `EVAL_BATCH_SIZE` | `32` | Batch size valutazione |
| `LEARNING_RATE` | `1e-3` | Learning rate (alto perché si allena solo la head) |
| `NUM_EPOCHS` | `10` | Epoche di training |
| `VALIDATION_SPLIT` | `0.2` | Frazione di validation split interno |
| `WEIGHT_DECAY` | `0.01` | Weight decay |
| `SEED` | `42` | Seed per riproducibilità |
| `HF_TOKEN` | _(vuoto)_ | Token HF — obbligatorio per `push_model.py` |

---

## Riferimento CLI completo

### export_feedback.py

```bash
python export_feedback.py <model_version> \
  [--date-from ISO8601] \
  [--date-to ISO8601] \
  [--limit N] \
  [--output-dir PATH]
```

### fine_tune.py

```bash
python fine_tune.py <candidate_version> \
  [--train-path PATH]        # default: GOLDEN_SET_DIR
  [--output-dir PATH]        # default: TRAIN_OUTPUT_DIR
  [--base-model NAME]        # default: HF_BASE_MODEL
  [--max-length N]           # default: MAX_LENGTH
  [--batch-size N]           # default: TRAIN_BATCH_SIZE
  [--epochs N]               # default: NUM_EPOCHS
  [--learning-rate F]        # default: LEARNING_RATE
  [--weight-decay F]         # default: WEIGHT_DECAY
  [--validation-split F]     # default: VALIDATION_SPLIT
  [--seed N]                 # default: SEED
```

### evaluate.py

```bash
python evaluate.py <candidate_version> \
  [--model-dir PATH]         # default: TRAIN_OUTPUT_DIR/<version>
  [--eval-set PATH]          # default: EVAL_SET_PATH
  [--output-dir PATH]        # default: EVAL_RESULTS_DIR
  [--batch-size N]           # default: EVAL_BATCH_SIZE
  [--max-length N]           # default: MAX_LENGTH
  [--report-only]            # non invia le metriche all'API
```

### push_model.py

```bash
python push_model.py <version> \
  [--model-dir PATH]         # default: TRAIN_OUTPUT_DIR/<version>
  [--repo-id ID]             # default: HF_REPO_ID
  [--create-tag]             # crea il tag <version> su HF dopo il push
  [--commit-message MSG]     # default: "Add fine-tuned candidate <version>"
```
