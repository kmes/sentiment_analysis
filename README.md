# sentiment-analysis

API FastAPI per l'analisi del sentiment (negativo / neutro / positivo) basata su RoBERTa, con pipeline MLOps completa per la raccolta di feedback, il fine-tuning del modello, la valutazione e la promozione in produzione.

---

## Indice

1. [Architettura](#architettura)
2. [Prerequisiti](#prerequisiti)
3. [Configurazione iniziale](#configurazione-iniziale)
4. [Avviare il progetto in locale](#avviare-il-progetto-in-locale)
5. [Caricare il modello](#caricare-il-modello)
6. [Endpoint API](#endpoint-api)
7. [Test di integrazione](#test-di-integrazione)
8. [CI/CD — GitHub Actions](#cicd--github-actions)
9. [Fine-tuning del modello](#fine-tuning-del-modello)

---

## Architettura

Il progetto è composto da più componenti orchestrati con Docker Compose:

| Componente | Tecnologia | Ruolo |
|---|---|---|
| **api** | FastAPI + Uvicorn | Inferenza sentiment, raccolta feedback, esposizione metriche |
| **postgres** | PostgreSQL 16 | Storage di predizioni, feedback e metriche modello |
| **prometheus** | Prometheus | Scraping metriche esposte dall'API *(profilo `monitoring`)* |
| **grafana** | Grafana Enterprise | Dashboard di osservabilità *(profilo `monitoring`)* |
| **airflow** | Apache Airflow 2.10 | DAG che ricalcola `feedback_disagreement_rate` ogni 5 minuti *(profilo `airflow`)* |
| **ml_workspace** | Python (no Docker) | Script per esportare feedback, fare fine-tuning e pubblicare nuove versioni del modello |

---

## Prerequisiti

- [Docker](https://docs.docker.com/get-docker/) e Docker Compose
- Python ≥ 3.11 — solo per usare `ml_workspace/` (fine-tuning, non richiesto per avviare l'API)

---

## Configurazione iniziale

Copiare il template delle variabili d'ambiente:

```bash
cp .env_sample .env
```

Le variabili hanno tutte valori di default funzionanti per un'istanza locale. Le più rilevanti:

| Variabile | Default | Descrizione |
|---|---|---|
| `APP_ENV` | `dev` | `dev` abilita gli endpoint di log diagnostici (`/logs/*`) |
| `HF_MODEL_NAME` | `frasem/sentiment-analysis-roberta` | Repository HuggingFace del modello |
| `HF_MODEL_REVISION` | `v1.0.0` | Tag o revision HF da scaricare all'avvio |
| `HF_MODEL_VERSION` | `v1.0.0` | Versione esposta da `/status` |
| `POSTGRES_USER` | `mlops` | Utente PostgreSQL |
| `POSTGRES_PASSWORD` | `mlops_secret` | Password PostgreSQL |
| `POSTGRES_DB` | `sentiment_db` | Nome del database |
| `GRAFANA_PASSWORD` | `admin` | Password Grafana *(profilo `monitoring`)* |
| `AIRFLOW_PASSWORD` | `admin` | Password Airflow *(profilo `airflow`)* |

---

## Avviare il progetto in locale

Tutti i comandi vanno eseguiti dalla root del repo, dove si trova `docker-compose.yml`.

**Base** — postgres + api:
```bash
docker compose up -d
```

**Con monitoring** — aggiunge Prometheus e Grafana:
```bash
docker compose --profile monitoring up -d
```

**Con Airflow** — aggiunge il DAG di refresh metriche:
```bash
docker compose --profile airflow up -d
```

**Tutto insieme**:
```bash
docker compose --profile monitoring --profile airflow up -d
```

Porte di accesso ai servizi:

| Servizio | URL | Profilo richiesto |
|---|---|---|
| API | http://localhost:8000 | sempre |
| Prometheus | http://localhost:9090 | `monitoring` |
| Grafana | http://localhost:3000 | `monitoring` |
| Airflow | http://localhost:8080 | `airflow` |

---

## Caricare il modello

Il modello non viene caricato automaticamente all'avvio: il container si avvia, ma le predizioni restituiranno HTTP 503 finché il modello non è in RAM.

Al **primo avvio** (o quando `HF_MODEL_REVISION` cambia), il modello viene scaricato automaticamente da HuggingFace prima che Uvicorn accetti richieste. Caricare il modello in memoria:

```bash
curl http://localhost:8000/load-model
```

Per scaricarlo e liberare la RAM:

```bash
curl http://localhost:8000/unload-model
```

---

## Endpoint API

### Ciclo di vita del modello

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `GET` | `/load-model` | Carica il modello in RAM |
| `GET` | `/unload-model` | Scarica il modello e libera la RAM |
| `GET` | `/status` | Health check: stato API, modello caricato, versione, uptime |

### Inferenza e feedback

Gli endpoint `/model/*` richiedono il modello caricato (HTTP 503 altrimenti), ad eccezione di `/model/feedback-export` e `/model/metrics`.

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `GET` | `/model/labels` | Restituisce le label valide: `negative`, `neutral`, `positive` |
| `POST` | `/model/predict` | Analisi del sentiment su testo libero |
| `POST` | `/model/predictions/{id}/feedback` | Salva la correzione dell'utente su una predizione esistente |
| `GET` | `/model/feedback-export` | Esporta i feedback raccolti per una versione del modello |
| `POST` | `/model/metrics` | Riceve le metriche di valutazione post-fine-tuning (HTTP 201) |

**POST /model/predict**

```json
// Richiesta
{ "text": "I really love this product!" }

// Risposta
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

**POST /model/predictions/{id}/feedback**

```json
// Richiesta — invia la label corretta per una predizione esistente
{ "label": "negative" }

// Risposta
{
  "status": "feedback received",
  "message": "Thank you for you feedback",
  "prediction_id": "550e8400-e29b-41d4-a716-446655440000",
  "label": "negative"
}
```

> Il feedback è idempotente lato risposta: HTTP 200 viene restituito anche se `prediction_id` non esiste o ha già un feedback associato.

**GET /model/feedback-export** — parametri query string:

| Parametro | Tipo | Obbligatorio | Descrizione |
|---|---|---|---|
| `model_version` | `str` | ✅ | Versione del modello (es. `v1.0.0`) |
| `date_from` | `datetime` ISO 8601 | ❌ | Inizio intervallo (clampato al minimo reale) |
| `date_to` | `datetime` ISO 8601 | ❌ | Fine intervallo (clampato al massimo reale) |
| `limit` | `int` | ❌ | Numero massimo di record; nessun limite se assente |

### Metriche e log diagnostici

| Metodo | Endpoint | Descrizione |
|---|---|---|
| `GET` | `/metrics` | Endpoint Prometheus (metriche in formato text/plain) |
| `GET` | `/internal/refresh-metrics` | Ricalcola `feedback_disagreement_rate` su finestra 24h (chiamato da Airflow) |
| `GET` | `/logs/predictions` | Lista predizioni paginata *(solo `APP_ENV=dev`)* |
| `GET` | `/logs/predictions/{id}` | Dettaglio predizione con eventuale feedback *(solo `APP_ENV=dev`)* |

---

## Test di integrazione

I test in `tests/` sono test di integrazione e richiedono API e postgres in esecuzione. Coprono:

- `test_01_db_status.py` — verifica che postgres risponda sulla porta configurata
- `test_02_api_status.py` — ciclo load/unload del modello e health check nei due stati
- `test_03_api_model.py` — flusso completo: predict → attesa persistenza DB → feedback → verifica log predizione

Per eseguirli localmente:

```bash
cp .env_sample .env
docker compose up -d
pip install -r tests/requirements.txt
pytest tests/ -v
```

---

## CI/CD — GitHub Actions

Il workflow [`.github/workflows/CI_CD.yml`](.github/workflows/CI_CD.yml) si attiva su ogni push al branch `main` ed esegue due job in sequenza.

### Job `test`

1. Avvia i container con `docker compose --env-file .env.ci up -d --build`
2. Attende che `GET /status` risponda (max 30 tentativi × 5 s)
3. Esegue `pytest tests/ -v`
4. Abbatte i container con `docker compose down -v`

### Job `build-and-push` *(eseguito solo se `test` passa)*

1. Effettua il login a GitHub Container Registry (`ghcr.io`)
2. Builda l'immagine Docker di `api/`
3. Pusha l'immagine con i tag `sha-<commit>` e `latest` su `ghcr.io/<owner>/<repo>/api`

### Testare il workflow in locale con `act`

[`act`](https://github.com/nektos/act) permette di eseguire i workflow GitHub Actions in locale tramite Docker, senza fare push su `main`.

Installazione:
```bash
# macOS / Linux (Homebrew)
brew install act

# Windows
winget install nektos.act
```

Al primo avvio seleziona l'immagine **Medium** (~500 MB), quella più compatibile con la maggior parte delle action.

Eseguire `act` senza argomenti tenta di eseguire entrambi i job (`test` e `build-and-push`). Per eseguire **solo il job `test`** (avvio container + pytest) senza richiedere credenziali ghcr.io:
```bash
act -j test
```

Per eseguire invece anche il job `build-and-push` è necessario un `GITHUB_TOKEN` valido per accedere a `ghcr.io`:
```bash
act -s GITHUB_TOKEN="$(gh auth token)"
```

Per elencare tutti i job disponibili nel progetto:
```bash
act -l
```

---

## Fine-tuning del modello

La cartella `ml_workspace/` contiene gli script per aggiornare il modello in produzione a partire dai feedback raccolti dall'API. Il flusso principale:

1. **Esporta i feedback** — `export_feedback.py` scarica dall'API i feedback per la versione corrente del modello
2. **Prepara il gold dataset** — il data scientist rivede i record e produce il dataset validato in `data/fine-tuning/`
3. **Fine-tuning** — `fine_tune.py` allena la classification head su RoBERTa congelato e salva il candidate in `runs/`
4. **Valutazione** — `evaluate.py` calcola accuracy e F1 sull'eval set e invia le metriche all'API
5. **Push su HuggingFace** — `push_model.py` pubblica il candidate approvato su HF e crea il tag di versione
6. **Deploy** — aggiornare `HF_MODEL_REVISION` nel `.env`, riavviare l'API con `docker compose up -d --build api` e chiamare `GET /load-model`

**Script di utilità** (opzionali, non obbligatori nel flusso principale):
- `sample_dataset.py` — campiona osservazioni bilanciate da un Parquet sorgente; utile per ridurre un dataset prima di passarlo a `simulate_traffic.py`
- `simulate_traffic.py` — simula utenti che effettuano predizioni e inviano feedback all'API; utile per popolare il DB senza aspettare traffico reale

→ Guida completa al fine-tuning: [ml_workspace/README.md](ml_workspace/README.md)
