# 🏦 Credit Risk Intelligence Agent

> An Agentic AI-powered credit risk assessment platform that combines machine learning, workflow orchestration, explainability, stress testing, and decision intelligence into a unified decision-making system.

Built using LangGraph, XGBoost, FastAPI, and Streamlit, this project explores how modern financial institutions can move beyond static credit scoring models and toward intelligent, explainable, and continuously evolving risk assessment systems.

---

## Why This Project Exists

Traditional credit risk assessment systems often suffer from three major limitations:

* Limited explainability
* Manual review bottlenecks
* Poor adaptability to changing economic conditions

Most credit scoring pipelines generate a prediction but provide little insight into why a decision was made or how risk changes under different market conditions.

This project explores an alternative approach:

Instead of treating credit assessment as a single prediction task, the system decomposes the problem into a network of specialized agents that collaborate to evaluate applicant risk, explain decisions, perform stress testing, and generate final recommendations.

---

## Key Capabilities

### Multi-Agent Credit Assessment Pipeline

The system uses specialized agents responsible for:

* Data preprocessing
* Risk scoring
* Explainability
* Stress testing
* Monitoring
* Decision generation

### Machine Learning Risk Prediction

Generates Probability of Default (PD) scores using trained machine learning models.

Current models include:

* XGBoost
* Logistic Regression
* Random Forest

### Explainable AI

Every decision is accompanied by human-readable explanations that help justify risk assessments.

### Stress Testing

Evaluates applicant resilience under simulated economic shocks.

### API-First Architecture

Credit decisions can be consumed programmatically through REST APIs.

### Interactive Dashboard

Streamlit interface enables quick experimentation and analysis.

---

# System Architecture

```text
                    ┌───────────────────┐
                    │ Applicant Data    │
                    └─────────┬─────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ Preprocessing Agent    │
                 └─────────┬──────────────┘
                           │
                           ▼
                 ┌────────────────────────┐
                 │ Risk Scoring Agent     │
                 └─────────┬──────────────┘
                           │
                           ▼
                 ┌────────────────────────┐
                 │ Explainability Agent   │
                 └─────────┬──────────────┘
                           │
                           ▼
                 ┌────────────────────────┐
                 │ Stress Testing Agent   │
                 └─────────┬──────────────┘
                           │
                           ▼
                 ┌────────────────────────┐
                 │ Monitoring Agent       │
                 └─────────┬──────────────┘
                           │
                           ▼
                 ┌────────────────────────┐
                 │ Decision Agent         │
                 └─────────┬──────────────┘
                           │
                           ▼
                    Final Recommendation
```

---

# LangGraph Workflow

The workflow is implemented as a LangGraph state machine.

```text
preprocessing
      ↓
risk_scoring
      ↓
explainability
      ↓
stress_test
      ↓
monitoring
      ↓
router
```

Each node contributes a specific piece of information to the evolving system state before passing control to the next stage.

---

# Agent Breakdown

## Preprocessing Agent

Responsibilities:

* Data validation
* Feature engineering
* Applicant normalization
* Missing value handling

Example engineered features:

* Debt-to-income ratio
* Credit utilization
* Employment encoding

---

## Risk Scoring Agent

Responsibilities:

* Generate Probability of Default
* Risk categorization
* Confidence estimation

Models:

* XGBoost
* Logistic Regression
* Random Forest

---

## Explainability Agent

Responsibilities:

* Translate model outputs into understandable explanations
* Surface major contributors to risk

Example outputs:

* High debt burden
* Low credit score
* Elevated utilization ratio

---

## Stress Testing Agent

Responsibilities:

* Simulate adverse economic conditions
* Recalculate risk exposure

Current implementation:

* 20% shock scenario

---

## Monitoring Agent

Responsibilities:

* System health monitoring
* Pipeline validation
* Workflow observability

---

## Decision Agent

Responsibilities:

Generate:

* APPROVE
* HOLD
* REJECT

based on aggregate workflow outputs.

---

# Machine Learning Pipeline

Dataset:

UCI Default of Credit Card Clients

Characteristics:

* ~30,000 records
* 24 features
* Binary default prediction target

Workflow:

Raw Data
→ Feature Engineering
→ Model Training
→ Evaluation
→ Best Model Selection
→ Deployment

Artifacts are automatically stored and reused by downstream agents.

---

# Tech Stack

## AI / ML

* Python
* XGBoost
* Scikit-Learn
* Pandas
* NumPy

## Agent Framework

* LangGraph

## Backend

* FastAPI
* Pydantic

## Frontend

* Streamlit

## Data Storage

* CSV datasets
* Serialized model artifacts

## API Layer

* REST APIs
* JSON schemas

---

# API Example

POST /analyze

```json
{
  "applicant_name": "Jane Doe",
  "income": 75000,
  "existing_debt": 15000,
  "credit_score": 680,
  "loan_amount": 20000
}
```

Response:

```json
{
  "decision": "APPROVE",
  "pd": 0.18,
  "risk_category": "low"
}
```

---

# Current Status

## Completed

* Multi-agent architecture
* LangGraph orchestration
* Credit risk prediction
* REST API
* Dashboard interface
* Stress testing
* Monitoring pipeline

## In Progress

* Enhanced explainability
* Monitoring improvements
* Expanded stress scenarios

## Planned

* Financial news sentiment analysis
* Portfolio-level risk management
* Regulatory policy retrieval
* Alternative data integration
* Fraud detection
* Continuous learning pipelines
* Real-time borrower monitoring
* Enterprise deployment architecture

---

# Future Vision

The long-term goal is to evolve this project from a credit scoring application into a Credit Risk Intelligence Platform.

Potential future modules include:

* Agentic credit analysts
* RAG-powered regulatory assistants
* Vector database integration
* Market intelligence ingestion
* Early warning systems
* Autonomous portfolio monitoring
* Human-in-the-loop review workflows
* Institution-facing APIs
* Cloud-native deployment
* Real-time decision engines

---

# Engineering Lessons

The most important lesson from this project was that building practical AI systems is rarely about the model itself.

Most complexity emerged from:

* Workflow orchestration
* State management
* Explainability
* Monitoring
* Validation
* Decision consistency

The project provided hands-on exposure to the challenges of designing reliable agentic systems that combine machine learning with structured decision-making workflows.

---

# Author

Amritpal Singh , Atul Kumr, Utkarsh Khatal, Akshita Joshi

B.Tech (Electronics & Computer Science)

Interests:
Agentic AI • Machine Learning • Full-Stack Development • Financial Intelligence Systems • Workflow Orchestration
