# Sentiment Analysis API — Specifiche

## Panoramica

Applicazione MLOps-oriented che espone una REST API per l'analisi del sentiment di testi in linguaggio naturale. Utilizza il modello `frasem/sentiment-analysis-roberta` (fork personale di RoBERTa fine-tuned su Twitter, originariamente basato su `cardiffnlp/twitter-roberta-base-sentiment-latest`) per classificare testi in tre categorie: **negative**, **neutral**, **positive**.

Ogni predizione viene registrata su PostgreSQL. Gli utenti possono inviare feedback sulle predizioni, rendendo possibile la raccolta dati per future sessioni di fine-tuning. Il progetto include uno spazio di lavoro dedicato al fine-tuning (`ml_workspace/`), un sistema di monitoring tramite Prometheus e Grafana, una pipeline CI/CD su GitHub Actions, e job periodici orchestrati con Apache Airflow.

---

## Tech Stack

| Area | Tecnologia |
|---|---|
| Linguaggio | Python 3.11 |
| Framework web | FastAPI 0.135.3, Uvicorn 0.44.0 (ASGI) |
| Validazione dati | Pydantic 2.13.0 |
| ML / NLP | Hugging Face Transformers 5.5.4, PyTorch 2.9.1 (CPU), safetensors 0.7.0 |
| Modello | `frasem/sentiment-analysis-roberta` (repo HF personale, RoBERTa-base) |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy 2.0.49 (async), asyncpg 0.31.0, psycopg2-binary 2.9.11, Alembic 1.18.4 |
| Monitoring | prometheus_client 0.23.1, Prometheus (container), Grafana Enterprise (container) |
| Orchestrazione job | Apache Airflow 2.10.4 (LocalExecutor, provider HTTP) |
| Logging | python-json-logger 4.0.0 (output JSON strutturato) |
| Containerizzazione | Docker, Docker Compose (con profiles) |
| CI/CD | GitHub Actions → GitHub Container Registry (`ghcr.io`) |
| Testing | pytest 9.0.3, requests 2.33.1, jsonschema 4.26.0 |

---

## Architettura

I container sono organizzati in tre livelli:

- **Core** (sempre attivi): `postgres`, `api`
- **Profilo `monitoring`**: `prometheus`, `grafana`
- **Profilo `airflow`**: `airflow-postgres`, `airflow-init`, `airflow-webserver`, `airflow-scheduler`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Docker Compose                                                               │
│                                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐                      │
│  │  PostgreSQL  │◄───│     API      │◄───│ Prometheus │  profilo: monitoring  │
│  │  (porta 5432)│    │  (porta 8000)│    │ (porta     │                      │
│  │              │    │              │    │  9090)     │                      │
│  │  4 tabelle:  │    │  FastAPI +   │    │            │                      │
│  │  - inference │    │  RoBERTa     │    │  scrape    │                      │
│  │  - feedback  │    │  in memoria  │    │  /metrics  │                      │
│  │  - model_load│    │  + /metrics  │    │  ogni 15s  │                      │
│  │  - model_eval│    │              │    └────────────┘                      │
│  └──────────────┘    └──────┬───────┘         │                             │
│                             │            ┌────▼───────┐                      │
│                             │            │  Grafana   │  profilo: monitoring  │
│                             │            │ (porta     │                      │
│  ┌──────────────┐           │            │  3000)     │                      │
│  │  airflow-    │           │            └────────────┘                      │
│  │  postgres    │           │                                                │
│  │  (porta 5433)│    ┌──────▼──────────────────────────────┐                │
│  └──────┬───────┘    │  Airflow           profilo: airflow  │                │
│         │            │                                      │                │
│  ┌──────▼───────┐    │  airflow-init  (run-once)            │                │
│  │  airflow-    │    │  airflow-webserver  (porta 8080)      │                │
│  │  init / ws / │    │  airflow-scheduler                   │                │
│  │  scheduler   │    │    → chiama GET /internal/refresh-   │                │
│  └──────────────┘    │      metrics ogni 5 min              │                │
│                      └──────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────┘
```

Il modello HF viene scaricato a runtime all'avvio dell'API (`download_mlmodel.py`) e salvato localmente in `api/ml_models/sentiment-analysis-roberta/`. La directory persiste tra restart e rebuild perché `api/` è esposta nel container come bind mount (`./api:/app` in `docker-compose.yml`). Il caricamento del modello a livello applicativo usa `local_files_only=True`, quindi il download preliminare è obbligatorio. Il check di revision usa un file lock `.hf_revision` nella directory del modello: se la revision già salvata corrisponde a `HF_MODEL_REVISION`, il download viene saltato.

L'entrypoint del container API (`api/Dockerfile`) accende `--reload` di Uvicorn solo quando `APP_ENV=dev`; in produzione il flag non viene aggiunto.

### Avvio

```bash
# Solo core (sviluppo/debug)
docker compose up -d

# Stack completo (sviluppo con monitoring e Airflow)
docker compose --profile monitoring --profile airflow up -d

# CI: il workflow usa docker compose --env-file .env.ci up -d --build
# senza --profile → avvia solo postgres e api
```

---

## Struttura del Progetto

```
sentiment_analysis/
├── api/
│   ├── app.py                    # Entry point Uvicorn
│   ├── boostrap.py               # Configurazione FastAPI + lifespan + mount /metrics
│   ├── dependencies.py           # Singleton: SentimentAnalyzer, TimerHelper, metriche Prometheus
│   ├── middlewares.py            # Guard: check_model_loaded(), require_dev_env()
│   ├── schemas.py                # Pydantic models (request/response)
│   ├── download_mlmodel.py       # Script pre-avvio: scarica il modello se assente o se la revision è cambiata
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── routes/
│   │   ├── __init__.py           # Aggregazione router
│   │   ├── status.py             # Health check
│   │   ├── load_model.py         # Carica/scarica modello in RAM
│   │   ├── model.py              # Predict + feedback + feedback-export + metrics
│   │   ├── internal.py           # Endpoint interni (refresh-metrics per Airflow)
│   │   └── logs.py               # Lettura log (solo dev)
│   ├── helpers/
│   │   ├── SentimentAnalyzer.py  # Wrapper pipeline HuggingFace
│   │   └── TimerHelper.py        # Misura latenze e uptime
│   ├── db/
│   │   ├── database.py           # Connessione async PostgreSQL
│   │   ├── models.py             # ORM SQLAlchemy (4 tabelle + indici)
│   │   └── services.py           # Query async (insert, select, paginate, export, metrics)
│   └── ml_models/                # Cache locale modello (volume Docker)
│       └── sentiment-analysis-roberta/
├── airflow/
│   └── dags/
│       └── refresh_metrics.py    # DAG: chiama /internal/refresh-metrics ogni 5 min
├── ml_workspace/                 # Area di lavoro fine-tuning (non dockerizzata)
│   ├── .env                      # Variabili d'ambiente locali (da .env_sample)
│   ├── .env_sample               # Template variabili d'ambiente
│   ├── requirements.txt
│   ├── README.md                 # Guida operativa del workspace
│   ├── common.py                 # Modulo condiviso: dotenv, label mapping, parsing JSONL, payload metriche
│   ├── export_feedback.py        # CLI (typer): esporta feedback da API → JSONL + manifest
│   ├── fine_tune.py              # CLI (typer): classifier-only fine-tuning, freeze backbone, output in runs/
│   ├── evaluate.py               # CLI (typer): valuta candidate su EVAL_SET_PATH, POST opzionale a /model/metrics
│   ├── push_model.py             # CLI (typer): pubblica candidate approvato su HF Hub + crea tag di versione
│   ├── sample_dataset.py         # CLI (typer): campiona dataset bilanciato da un parquet sorgente
│   ├── simulate_traffic.py       # CLI (typer): emula traffico predict+feedback verso l'API
│   ├── tests/                    # Test unitari pytest (parsing, label mapping, payload, posting HTTP)
│   ├── model/                    # [submodule] frasem/sentiment-analysis-roberta (modello base)
│   ├── runs/                     # Output di fine_tune.py: candidate models versionati (creato on-demand)
│   ├── artifacts/
│   │   └── evaluations/          # Output di evaluate.py: report JSON per candidate version
│   └── data/
│       ├── raw/                  # Bronze: output grezzo di export_feedback.py (JSONL + manifest)
│       ├── silver/               # Silver: dati filtrati/puliti prima della validazione
│       ├── fine-tuning/          # Golden: dataset validato usato per il training (cumulativo)
│       ├── evaluate/             # Eval set Parquet per evaluate.py (data/evaluate/eval.parquet)
│       ├── simulation/           # Dataset per simulate_traffic.py (sorgente + sample bilanciato)
│       └── sentiment-dataset/    # [submodule] frasem/sentiment-dataset — sorgente Parquet train/test/validation
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml
│       └── dashboards/
│           ├── dashboards.yml
│           └── sentiment_api.json  # Dashboard JSON (9 panel)
├── tests/
│   ├── test_01_db_status.py
│   ├── test_02_api_status.py
│   ├── test_03_api_model.py
│   └── helper/
│       ├── __init__.py
│       └── TestEndpoint.py
├── .github/workflows/CI_CD.yml
├── docker-compose.yml
├── .env
├── .env.ci
└── .env_sample
```

---

## Funzionalità e Endpoint API

### Ciclo di vita del modello

Il modello non viene caricato automaticamente all'avvio: deve essere caricato esplicitamente.

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/load-model` | GET | Carica il modello RoBERTa in RAM, registra `model_load_time_ms` |
| `/unload-model` | GET | Scarica il modello e libera la RAM |
| `/status` | GET | Health check: stato API, modello caricato, load_id, uptime, versione |

### Inferenza e feedback

Gli endpoint `/model/*` richiedono che il modello sia caricato (middleware `check_model_loaded` → HTTP 503 altrimenti), ad eccezione di `/model/feedback-export` e `/model/metrics` che usano un router separato senza il middleware.

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/model/labels` | GET | Restituisce le label valide: `["negative", "neutral", "positive"]` |
| `/model/predict` | POST | Inferenza sentiment su testo libero |
| `/model/predictions/{id}/feedback` | POST | Salva la correzione dell'utente su una predizione esistente |
| `/model/train` | POST | Placeholder non implementato |
| `/model/feedback-export` | GET | Esporta feedback per versione modello (per fine-tuning) |
| `/model/metrics` | POST | Riceve metriche di valutazione post-fine-tuning (HTTP 201) |

**Richiesta predict:**
```json
{ "text": "I really love this product!" }
```

**Risposta predict:**
```json
{
  "status": "predicted",
  "message": "Result positive with 1.00 of confidence",
  "input_text": "I really love this product!",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_load_id": "...",
  "predicted_label": "positive",
  "confidence": 0.9987,
  "latency_ms": 45
}
```

Il log della predizione viene salvato in background. Ad ogni predizione vengono aggiornate le metriche `prediction_latency_ms`, `predictions_total`, `prediction_text_length`, `prediction_confidence`.

Il feedback è idempotente lato risposta: viene sempre restituito HTTP 200 anche se `prediction_id` non esiste o ha già un feedback.

#### GET /model/feedback-export

Parametri query string:

| Parametro | Tipo | Obbligatorio | Note |
|---|---|---|---|
| `model_version` | `str` | ✅ | |
| `date_from` | `datetime` ISO 8601 | ❌ | Clampato al MIN reale per la revision |
| `date_to` | `datetime` ISO 8601 | ❌ | Clampato al MAX reale per la revision |
| `limit` | `int` | ❌ | Nessun cap se assente |

Risposta: `FeedbackExportResponse` con `records` (lista di `FeedbackExportRecord`), date effettive, totale record.

#### POST /model/metrics

Riceve `ModelMetricsPayload` (accuracy, f1_macro, f1_negative/neutral/positive, eval_loss, num_samples, model_version, eval_dataset). Persiste su `model_evaluation_logs` e aggiorna le Gauge Prometheus `model_accuracy`, `model_f1_macro`, `model_f1_per_label`.

### Endpoint interni

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/internal/refresh-metrics` | GET | Ricalcola `feedback_disagreement_rate` (finestra 24h) e aggiorna la Gauge Prometheus |

Chiamato dal DAG Airflow ogni 5 minuti. Non ha middleware di protezione.

### Metriche (Prometheus)

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/metrics` | GET | Endpoint Prometheus, esposto tramite `make_asgi_app()` |

### Log diagnostici (solo `APP_ENV=dev`)

Protetti dal middleware `require_dev_env` (→ HTTP 403 in produzione).

| Endpoint | Metodo | Descrizione |
|---|---|---|
| `/logs/loaded-model` | GET | Log dei caricamenti modello (paginati) |
| `/logs/predictions` | GET | Lista predizioni (paginata, filtro opzionale `only_with_feedback`) |
| `/logs/predictions/{id}` | GET | Singola predizione con feedback associato |

---

## Database — Schema

```
model_load_logs
├── id                 (PK, integer, auto-increment)
├── model_load_id      (UUID, unique)
├── model_name         (string 255)
├── model_version      (string 50)
├── load_time_ms       (integer)
├── timestamp          (datetime con timezone, default now())
└── [idx_model_load_timestamp, idx_model_load_id]

inference_logs
├── id                 (PK, integer, auto-increment)
├── prediction_id      (UUID, unique)
├── model_load_id      (FK → model_load_logs.model_load_id)
├── input_text         (text)
├── predicted_label    (string 20)
├── confidence         (float)
├── latency_ms         (integer)
├── timestamp          (datetime con timezone, default now())
├── relationship: feedback (1:1 → feedback_logs)
└── [idx_inference_timestamp, idx_inference_prediction_id, idx_inference_model_load_id]

feedback_logs
├── id                 (PK, integer, auto-increment)
├── prediction_id      (FK → inference_logs.prediction_id, unique)
├── true_label         (string 20)
├── timestamp          (datetime con timezone, default now())
└── [idx_feedback_prediction_id, idx_feedback_timestamp]

model_evaluation_logs
├── id                 (PK, integer, auto-increment)
├── model_version      (string 50)
├── eval_dataset       (string 255)
├── accuracy           (float)
├── f1_macro           (float)
├── f1_negative        (float)
├── f1_neutral         (float)
├── f1_positive        (float)
├── eval_loss          (float)
├── num_samples        (integer)
└── timestamp          (datetime con timezone, default now())
```

Le tabelle vengono create automaticamente all'avvio se non esistono (idempotente, via `Base.metadata.create_all`).

---

## Metriche Prometheus

Undici metriche custom definite in `dependencies.py`:

| Nome | Tipo | Label | Descrizione |
|---|---|---|---|
| `prediction_latency_ms` | Histogram | — | Latenza di ogni `/predict` in ms. Bucket: [10, 25, 50, 100, 250, 500, 1000] |
| `model_load_time_ms` | Histogram | — | Tempo di caricamento modello in ms. Bucket: [500, 1000, 2000, 5000, 10000] |
| `predictions_total` | Counter | `predicted_label` | Contatore cumulativo predizioni per label |
| `model_loaded` | Gauge | — | Stato corrente modello: 1 = caricato, 0 = scaricato |
| `prediction_text_length_chars` | Histogram | — | Lunghezza testo di input in caratteri. Bucket: [10, 25, 50, 100, 200, 500, 1000, 2000] |
| `prediction_confidence` | Histogram | — | Confidence score delle predizioni. Bucket: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0] |
| `feedback_disagreement_rate` | Gauge | — | % feedback con `true_label != predicted_label` (finestra rolling 24h, aggiornata da Airflow) |
| `feedback_total` | Counter | `true_label` | Contatore feedback ricevuti per label |
| `model_accuracy` | Gauge | `model_version` | Accuracy dell'ultima valutazione per versione |
| `model_f1_macro` | Gauge | `model_version` | F1 macro dell'ultima valutazione per versione |
| `model_f1_per_label` | Gauge | `label`, `model_version` | F1 per label (negative/neutral/positive) per versione |

Prometheus scrapa `/metrics` ogni 15 secondi.

---

## Dashboard Grafana

Dashboard auto-provisioned (`grafana/provisioning/dashboards/sentiment_api.json`) con 9 panel:

| Panel | Tipo | Metrica / Query |
|---|---|---|
| **Modello caricato** | Gauge | `model_loaded` |
| **Predizioni al secondo (per label)** | TimeSeries | `rate(predictions_total[5m])` per `predicted_label` |
| **Latenza predizioni (p50/p95/p99)** | TimeSeries | `histogram_quantile(0.50/0.95/0.99, rate(prediction_latency_ms_bucket[5m]))` |
| **Tempo caricamento modello (p50/p95)** | TimeSeries | `histogram_quantile(0.50/0.95, rate(model_load_time_ms_bucket[5m]))` |
| **Feedback disagreement rate** | TimeSeries | `feedback_disagreement_rate` |
| **Confidence distribution (p50/p95)** | TimeSeries | `histogram_quantile(0.50/0.95, rate(prediction_confidence_bucket[5m]))` |
| **Text length distribution (p50/p95)** | TimeSeries | `histogram_quantile(0.50/0.95, rate(prediction_text_length_chars_bucket[5m]))` |
| **Feedback per label** | TimeSeries | `rate(feedback_total[5m])` per `true_label` |
| **Model quality per version** | BarGauge | `model_accuracy`, `model_f1_macro`, `model_f1_per_label` con `model_version` come dimensione |

Refresh automatico: 30s. Intervallo temporale default: ultima ora.

---

## Apache Airflow

Tre container con profilo `airflow`, con database PostgreSQL dedicato (`airflow-postgres`, porta host 5433).

| Container | Ruolo |
|---|---|
| `airflow-postgres` | Database metadata Airflow (PostgreSQL 16, database `airflow`) |
| `airflow-init` | Esegue migration DB e crea utente admin (`command: version`, `restart: "no"`) |
| `airflow-webserver` | UI Airflow (porta 8080) |
| `airflow-scheduler` | Esegue i DAG con LocalExecutor |

### DAG: refresh_sentiment_metrics

File: `airflow/dags/refresh_metrics.py`

- Schedule: `*/5 * * * *` (ogni 5 minuti)
- Task: `refresh_disagreement_rate` (`SimpleHttpOperator`) → `GET /internal/refresh-metrics` tramite connessione `sentiment_api` (`http_conn_id`)
- Connessione configurata via variabile d'ambiente `AIRFLOW_CONN_SENTIMENT_API` nel container scheduler (host `http://api:8000`)
- Il provider HTTP non è incluso nell'immagine ufficiale: viene installato a runtime in webserver e scheduler tramite `_PIP_ADDITIONAL_REQUIREMENTS=apache-airflow-providers-http`

---

## ml_workspace — Area Fine-Tuning

```
ml_workspace/
├── .env                   # Variabili d'ambiente locali (caricate via python-dotenv)
├── requirements.txt       # huggingface_hub, typer, httpx, transformers, torch, datasets, sklearn, etc.
├── common.py              # Modulo condiviso: dotenv, label mapping, parsing JSONL, payload metriche
├── export_feedback.py     # CLI: esporta feedback da API per il fine-tuning
├── fine_tune.py           # CLI: classifier-only fine-tuning su gold dataset cumulativo
├── evaluate.py            # CLI: valuta candidate model + POST opzionale a /model/metrics
├── push_model.py          # CLI: pubblica candidate approvato su HF Hub + crea tag di versione
├── sample_dataset.py      # CLI: campiona dataset bilanciato da un parquet sorgente
├── simulate_traffic.py    # CLI: emula traffico predict+feedback verso l'API
├── tests/                 # Test pytest unitari (parsing, label mapping, payload, posting HTTP)
├── model/                 # Clone git di frasem/sentiment-analysis-roberta
├── runs/                  # Candidate models prodotti da fine_tune.py; push_model.py pubblica da qui su HF (model/ è il modello sorgente, non viene mai toccato)
├── artifacts/
│   └── evaluations/       # Report JSON di evaluate.py per candidate version
└── data/
    ├── raw/               # Bronze: output grezzo di export_feedback.py
    │   ├── feedback_<version>_<timestamp>.jsonl
    │   └── manifest_<version>_<timestamp>.json
    ├── silver/            # Silver: dati filtrati/puliti prima della validazione
    ├── fine-tuning/       # Golden: dataset cumulativo validato usato per il training
    ├── evaluate/          # Eval set Parquet per evaluate.py (data/evaluate/eval.parquet)
    ├── simulation/        # Dataset per simulate_traffic.py (sorgente + sample bilanciato)
    └── sentiment-dataset/ # [submodule] frasem/sentiment-dataset — sorgente Parquet train/test/validation
```

### Variabili d'ambiente (`ml_workspace/.env`)

| Variabile | Default | Descrizione |
|---|---|---|
| `SENTIMENT_API_URL` | `http://localhost:8000` | URL base dell'API |
| `FEEDBACK_EXPORT_ENDPOINT` | `/model/feedback-export` | Endpoint export feedback |
| `BRONZE_SET_DIR` | `data/raw` | Directory output di `export_feedback.py` |
| `SILVER_SET_DIR` | `data/silver` | Directory dataset silver |
| `GOLDEN_SET_DIR` | `data/fine-tuning` | Directory dataset golden per training |
| `EVAL_SET_PATH` | `data/evaluate/eval.parquet` | Path dataset di valutazione |
| `MODEL_PATH` | `model` | Path del clone locale del modello |
| `HF_BASE_MODEL` | `frasem/sentiment-analysis-roberta` | Modello base da cui partire per il fine-tuning |
| `HF_REPO_ID` | `frasem/sentiment-analysis-roberta` | Repo HF di destinazione per push (`push_model.py`) |
| `TRAIN_OUTPUT_DIR` | `runs` | Directory dei candidate models prodotti da `fine_tune.py` |
| `EVAL_RESULTS_DIR` | `artifacts/evaluations` | Directory dei report di `evaluate.py` |
| `MAX_LENGTH` | `256` | Lunghezza massima tokenizzazione |
| `TRAIN_BATCH_SIZE` | `16` | Batch size per il training |
| `EVAL_BATCH_SIZE` | `32` | Batch size per la valutazione |
| `LEARNING_RATE` | `1e-3` | Learning rate (alto perché si allena solo la head) |
| `NUM_EPOCHS` | `10` | Numero epoche di training |
| `VALIDATION_SPLIT` | `0.2` | Frazione di validation split interno (non usa `EVAL_SET_PATH`) |
| `WEIGHT_DECAY` | `0.01` | Weight decay |
| `SEED` | `42` | Seed per riproducibilità |
| `HF_TOKEN` | _vuoto_ | Token HF per autenticazione, usato da `push_model.py` |

> **Submodule git**: `model/` e `data/sentiment-dataset/` sono git submoduli. Dopo il clone del repo inizializzarli con: `git submodule update --init --recursive`

### export_feedback.py

Script CLI (typer) che chiama `GET /model/feedback-export` e salva i risultati in `data/raw/` (configurabile tramite `BRONZE_SET_DIR` nel `.env` o `--output-dir`). Le variabili d'ambiente vengono caricate automaticamente da `ml_workspace/.env` tramite `python-dotenv`.

```bash
python export_feedback.py <model_version> [--date-from ISO] [--date-to ISO] [--limit N] [--output-dir PATH]
```

Output JSONL: ogni riga `{"text": ..., "label": ..., "_predicted_label": ..., "_confidence": ..., "_prediction_id": ...}`. I campi con prefisso `_` sono metadati di review, non entrano nel training.

### fine_tune.py

Script CLI (typer) che esegue **classifier-only fine-tuning** sul modello base `HF_BASE_MODEL`: carica il gold dataset (cumulativo per default da `GOLDEN_SET_DIR`), congela il backbone RoBERTa (`model.roberta`) e allena solo la classification head. Usa HuggingFace `Trainer`, validation split interno (`VALIDATION_SPLIT`), seed fisso, CPU-oriented.

```bash
python fine_tune.py <candidate_version> [--train-path PATH] [--output-dir PATH] \
  [--max-length 256] [--batch-size 16] [--epochs 10] [--learning-rate 1e-3] \
  [--validation-split 0.2] [--seed 42]
```

Output: artefatti modello (`config.json`, `model.safetensors`, tokenizer files) in `runs/<candidate_version>/`, più `training_summary.json` con dataset, iperparametri, metriche di validation interna e timestamp. Il clone locale `ml_workspace/model/` non viene mai sovrascritto. Nessuna chiamata all'API, nessun push HF.

### evaluate.py

Script CLI (typer) che valuta un candidate model sull'eval set Parquet e (opzionalmente) invia le metriche a `POST /model/metrics`. Calcola `accuracy`, `f1_macro`, `f1_negative/neutral/positive`, `eval_loss` e `num_samples`. Salva il report JSON in `artifacts/evaluations/<candidate_version>.json` **anche in caso di errore HTTP** verso l'API.

```bash
python evaluate.py <candidate_version> [--model-dir PATH] [--eval-set PATH] [--report-only]
```

Default: `--model-dir` derivato da `TRAIN_OUTPUT_DIR/<candidate_version>`, `--eval-set` da `EVAL_SET_PATH`. Con `--report-only` non viene tentata la POST.

### push_model.py

Script CLI (typer) per promuovere un candidate model approvato dal `runs/` locale verso il repo HF (`HF_REPO_ID`). Richiede `HF_TOKEN` configurato in `.env`. Verifica la presenza di `config.json` nella model-dir prima di eseguire il push. Dopo l'upload, può creare opzionalmente un tag di versione nel repo.

```bash
python push_model.py <version> [--model-dir PATH] [--repo-id ID] [--create-tag] [--commit-message MSG]
```

Al termine stampa i passi successivi: aggiornare `HF_MODEL_REVISION=<version>` nel `.env` del progetto, riavviare l'API e chiamare `GET /load-model`.

### sample_dataset.py

Script CLI (typer) che legge un parquet sorgente, campiona N osservazioni con classi label bilanciate, e scrive un nuovo parquet pronto per `simulate_traffic.py`. Output sempre in formato Parquet con colonne `text` (str) e `label` (str). Normalizza la colonna label sorgente gestendo i tre casi: `label_text` (str), `label` (int → mappato via `ID2LABEL`), `label` (str). Se il numero richiesto supera il totale disponibile, estrae tutto.

```bash
python sample_dataset.py [--from-dataset PATH] [--to-dataset PATH] [--records N] [--seed S]
```

Default: `--from-dataset data/simulation/train-00000-of-00001.parquet`, `--to-dataset data/simulation/sample.parquet`, `--records 200`, `--seed` da `SEED` nel `.env`.

### simulate_traffic.py

Script CLI (typer) che emula traffico predict+feedback verso l'API a partire da un dataset etichettato. Per ogni record:

1. Verifica modello caricato via `GET /status`; se non lo è, chiama `GET /load-model` e ricontrolla
2. `POST /model/predict` con il testo del record → ottiene `prediction_id`
3. Verifica persistenza in DB con retry backoff lineare (10 tentativi, da 10 ms a 100 ms con incremento di 10 ms) via `GET /logs/predictions/{id}`; se la persistenza non avviene entro i tentativi, salta il record
4. `POST /model/predictions/{id}/feedback` con la `true_label` del dataset

```bash
python simulate_traffic.py [DATASET_PATH] [--api-url URL] [--limit N] [--delay S]
```

Default: `DATASET_PATH=data/simulation/sample.parquet`. Supporta JSONL (singolo file o directory) e Parquet, auto-detect dall'estensione. L'endpoint `GET /logs/predictions/{id}` è dev-only (richiede `APP_ENV=dev`). Pensato per popolare il DB con coppie predict+feedback riproducibili usate poi nel ciclo di fine-tuning.

### Flusso end-to-end

Il flusso operativo manuale per una sessione di fine-tuning è:

```bash
cd ml_workspace
python export_feedback.py v1.0.0
# fase data scientist esterna: produce/aggiorna gold dataset in data/fine-tuning/
python fine_tune.py v1.1.0-rc1 --train-path data/fine-tuning
python evaluate.py v1.1.0-rc1 --model-dir runs/v1.1.0-rc1
# revisione umana del report artifacts/evaluations/v1.1.0-rc1.json; se approvato:
python push_model.py v1.1.0 --model-dir runs/v1.1.0-rc1 --create-tag
# poi a livello di deploy:
# aggiornare HF_MODEL_REVISION=v1.1.0 nel .env del progetto
# docker compose up -d --build api
# curl http://localhost:8000/load-model
```

Airflow non è coinvolto in questo flusso; la promozione del modello in produzione resta sempre manuale.

---

## Configurazione

Variabili d'ambiente (`.env` / `.env.ci` / `.env_sample`):

| Variabile | Default | Descrizione |
|---|---|---|
| `APP_ENV` | `dev` | `dev` o `prod` — controlla accesso agli endpoint `/logs/*` |
| `POSTGRES_USER` | `mlops` | Credenziali DB API |
| `POSTGRES_PASSWORD` | `mlops_secret` | Credenziali DB API |
| `POSTGRES_DB` | `sentiment_db` | Nome database API |
| `POSTGRES_HOST` | `postgres` | Host database API |
| `POSTGRES_PORT` | `5432` | Porta database API |
| `API_PORT` | `8000` | Porta API |
| `HF_MODEL_NAME` | `frasem/sentiment-analysis-roberta` | Repo HF del modello |
| `HF_MODEL_SAVE_PATH` | `ml_models/sentiment-analysis-roberta` | Path locale cache modello |
| `HF_MODEL_VERSION` | `v1.0.0` | Tag versione modello |
| `HF_MODEL_REVISION` | `v1.0.0` | Revision HF; usata da `download_mlmodel.py` per il revision check |
| `GRAFANA_PORT` | `3000` | Porta Grafana |
| `GRAFANA_PASSWORD` | `admin` | Password admin Grafana |
| `AIRFLOW_PORT` | `8080` | Porta webserver Airflow |
| `AIRFLOW_PASSWORD` | `admin` | Password admin Airflow |
| `AIRFLOW_POSTGRES_USER` | `airflow` | Credenziali DB Airflow |
| `AIRFLOW_POSTGRES_PASSWORD` | `airflow_secret` | Credenziali DB Airflow |
| `AIRFLOW_POSTGRES_DB` | `airflow` | Nome database Airflow |
| `AIRFLOW_POSTGRES_PORT` | `5433` | Porta host DB Airflow (interno: 5432) |

`.env.ci` include solo le variabili core (`APP_ENV`, `POSTGRES_*`, `API_PORT`, `HF_*`). Grafana e Airflow non vengono avviati in CI.

---

## CI/CD

Pipeline GitHub Actions (`.github/workflows/CI_CD.yml`) attivata su push a `main`:

1. **Job `test`**
   - Carica variabili da `.env.ci` in `$GITHUB_ENV`
   - Setup Python 3.11 con cache pip
   - Build dei container con `docker compose --env-file .env.ci` (senza `--profile` → avvia solo `api` e `postgres`)
   - Attende che l'API risponda su `/status` (retry 30 volte)
   - Installa dipendenze test e esegue `pytest tests/ -v`
   - Teardown container

2. **Job `build-and-push`** (dipende da `test`)
   - Login a `ghcr.io`
   - Build + push immagine API con tag `sha-<commit>` e `latest`
   - Layer cache tramite GitHub Actions cache

---

## Avvio Rapido

```bash
# Configura
cp .env_sample .env

# Stack completo
docker compose --profile monitoring --profile airflow up -d --build

# Solo core (API + DB)
docker compose up -d --build

# API:         http://localhost:8000
# Prometheus:  http://localhost:9090
# Grafana:     http://localhost:3000  (admin / GRAFANA_PASSWORD)
# Airflow:     http://localhost:8080  (admin / AIRFLOW_PASSWORD)

# Caricare il modello
curl http://localhost:8000/load-model

# Predizione
curl -X POST http://localhost:8000/model/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This is amazing!"}'

# Feedback
curl -X POST http://localhost:8000/model/predictions/<prediction_id>/feedback \
     -H "Content-Type: application/json" \
     -d '{"label": "positive"}'

# Esportare feedback per fine-tuning (default output: data/raw da BRONZE_SET_DIR)
cd ml_workspace
python export_feedback.py v1.0.0

# Eseguire i test
pip install -r tests/requirements.txt
pytest tests/ -v
```
