# 🏦 Credit Risk Intelligence Agent

An agentic credit risk assessment system powered by **LangGraph**, **XGBoost**, and **Streamlit**. The system runs applicant data through a multi-step AI pipeline — preprocessing → risk scoring → explainability → stress testing → decision — and exposes both a web UI and a REST API.

---

## 📁 Project Structure

```
CreditRiskAgentt/
│
├── CreditRiskAgent/
│   ├── app.py                    # Streamlit UI (entry point)
│   ├── main.py                   # CLI runner
│   ├── requirements.txt
│   │
│   ├── data/
│   │   ├── raw/
│   │   │   └── uci_credit_card.csv
│   │   └── artifacts/
│   │       ├── xgboost.pkl
│   │       ├── logreg_baseline.pkl
│   │       ├── random_forest.pkl
│   │       ├── best_model_meta.json
│   │       └── model_metrics.csv
│   │
│   └── src/
│       ├── agents/
│       │   ├── preprocessing_agent.py
│       │   ├── risk_scoring_agent.py
│       │   ├── explainability_agent.py
│       │   ├── stress_test_agent.py
│       │   ├── monitoring_agent.py
│       │   └── decision_agent.py
│       ├── api/
│       │   ├── app.py            # FastAPI app
│       │   ├── logic.py
│       │   └── schemas.py
│       ├── graph/
│       │   ├── workflow.py       # LangGraph pipeline
│       │   ├── state.py
│       │   └── router.py
│       └── ml/
│           ├── train.py
│           └── predict.py
│
├── pipeline_service.py           # Standalone batch pipeline
├── pipeline_schemas.py
├── preprocessing_agent.py
├── risk_scoring_agent.py
├── run_pipeline.py               # Batch pipeline entry point
└── UCI_Credit_Card.csv
```

---

## ⚙️ How It Works

The core of the system is a **LangGraph state machine** with six sequential nodes:

```
preprocessing → risk_scoring → explainability → stress_test → monitoring → router
```

| Node | What it does |
|---|---|
| `preprocessing` | Engineers features (debt-to-income, credit utilization, employment encoding) |
| `risk_scoring` | Runs the trained XGBoost model to produce a Probability of Default (PD) score |
| `explainability` | Generates human-readable reasons for the risk score |
| `stress_test` | Simulates a 20% shock scenario on the base PD |
| `monitoring` | Reports system health status |
| `router` | Issues a final decision: **APPROVE / HOLD / REJECT** |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
cd CreditRiskAgent
pip install -r requirements.txt
```

### 2. Train the model (optional — pre-trained artifacts are included)

```bash
python src/ml/train.py
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

### 4. Run the FastAPI backend

```bash
uvicorn src.api.app:app --reload
```

API docs available at `http://localhost:8000/docs`

### 5. Run the batch pipeline (UCI dataset)

```bash
python run_pipeline.py
```

---

## 🔌 API Reference

### `POST /analyze`

Assess credit risk for a single applicant.

**Request body:**
```json
{
  "applicant_name": "Jane Doe",
  "age": 30,
  "income": 75000,
  "existing_debt": 15000,
  "credit_score": 680,
  "loan_amount": 20000,
  "employment_status": "employed"
}
```

**Response:**
```json
{
  "applicant_name": "Jane Doe",
  "decision": "APPROVE",
  "confidence": 0.9,
  "pd": 0.18,
  "risk_category": "low",
  "explanation": "Stable financial profile",
  "stress_results": {
    "base": 0.18,
    "shock_20pct": 0.216
  }
}
```

---

## 🧠 ML Models

Three models are trained and benchmarked; the best performer is selected automatically:

| Model | Notes |
|---|---|
| **XGBoost** | Primary model, selected by default |
| **Logistic Regression** | Baseline |
| **Random Forest** | Ensemble alternative |

Model artifacts and metrics are stored in `data/artifacts/`.

---

## 📊 Dataset

The system uses the [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) dataset (30,000 records, 24 features). Target variable: `default.payment.next.month`.

---

## 🛠 Tech Stack

- **LangGraph** — agentic workflow orchestration
- **XGBoost / scikit-learn** — ML models
- **Streamlit** — interactive web UI
- **FastAPI** — REST API
- **pandas / NumPy** — data processing
- **joblib** — model serialization