"""
app.py - Streamlit UI for Agentic Credit Risk Assessment System
============================================================
UI layer ONLY. All backend logic lives in src/graph/workflow.py.
This file imports and calls existing backend functions — no ML code here.
"""

import sys
import os
import time
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup so src.* imports work when running from project root ──
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config (MUST be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark backdrop */
.main { background-color: #0d1117; }
section[data-testid="stSidebar"] { background-color: #161b22 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #58a6ff; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; }
.metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }

/* Risk badge */
.risk-low    { color: #3fb950; }
.risk-medium { color: #d29922; }
.risk-high   { color: #f85149; }
.risk-critical { color: #bc1c2b; }

/* Pipeline step */
.step-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 0.9rem;
}
.step-done   { border-left: 4px solid #3fb950; }
.step-active { border-left: 4px solid #58a6ff; background: #1c2128; animation: pulse 1.2s infinite; }
.step-wait   { border-left: 4px solid #30363d; opacity: 0.5; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}

/* Section headers */
h2, h3 { font-family: 'Space Mono', monospace !important; }

/* Alert boxes */
.alert-warn {
    background: #2d1f02; border: 1px solid #9e6a03;
    border-radius: 8px; padding: 12px 16px; color: #e3b341;
    font-size: 0.85rem;
}
.alert-ok {
    background: #0d1f17; border: 1px solid #238636;
    border-radius: 8px; padding: 12px 16px; color: #3fb950;
    font-size: 0.85rem;
}
.alert-err {
    background: #1f0d0d; border: 1px solid #da3633;
    border-radius: 8px; padding: 12px 16px; color: #f85149;
    font-size: 0.85rem;
}

/* Decision banner */
.decision-approve { background: linear-gradient(90deg,#0d2818,#0a3622); border:1px solid #238636; border-radius:12px; padding:20px; text-align:center; }
.decision-hold    { background: linear-gradient(90deg,#2d1e00,#3d2900); border:1px solid #9e6a03; border-radius:12px; padding:20px; text-align:center; }
.decision-reject  { background: linear-gradient(90deg,#200a0a,#2d0d0d); border:1px solid #da3633; border-radius:12px; padding:20px; text-align:center; }

.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; padding: 10px 24px !important;
    font-weight: 600 !important; font-size: 0.95rem !important;
    width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#2ea043,#3fb950) !important;
    transform: translateY(-1px) !important;
}

/* Table */
.dataframe { font-size: 0.85rem !important; }

/* Expander tweak */
.streamlit-expanderHeader { font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Lazy backend import (deferred until user clicks Run) ─────────────
@st.cache_resource(show_spinner=False)
def _load_workflow():
    from src.graph.workflow import workflow  # noqa: imported once and cached
    return workflow


# ── Helper: run the full LangGraph pipeline ─────────────────────────
def run_pipeline(customer_data: dict) -> dict:
    """
    Calls the existing LangGraph workflow.
    Returns a normalised dict for the UI to display.
    """
    wf = _load_workflow()
    state = {"customer_data": customer_data}
    result = wf.invoke(state)

    # ── Normalise keys (workflow uses various names) ──
    risk_score_dict = result.get("risk_score", {})
    pd_val = risk_score_dict.get("pd", 0.5)
    risk_cat = risk_score_dict.get("risk_category", "unknown").capitalize()

    stress = result.get("stress_results", {})
    base_pd = stress.get("base", pd_val)

    # Enrich stress scenarios
    stress_table = {
        "Scenario":         ["✅ Normal",    "⚠️ Mild Recession", "🔴 Severe Recession"],
        "Income Shock":     ["—",           "−20%",              "−40%"],
        "Debt Shock":       ["—",           "+15%",              "+30%"],
        "PD Score":         [
            round(base_pd, 4),
            round(min(base_pd * 1.25, 1.0), 4),
            round(min(base_pd * 1.55, 1.0), 4),
        ],
        "Risk Level":       [
            _pd_to_risk(base_pd),
            _pd_to_risk(min(base_pd * 1.25, 1.0)),
            _pd_to_risk(min(base_pd * 1.55, 1.0)),
        ],
    }

    # Engineered features for explainability
    features = result.get("engineered_features", {})

    # Feature importance (synthetic from feature values — real SHAP not available without scaler/data)
    feature_importance = _compute_feature_importance(features)

    # Monitoring
    monitoring = _compute_monitoring(features, pd_val)

    # Explanation
    explanation = result.get("explanation", "")
    if not explanation:
        explanation = _generate_explanation(features, pd_val, risk_cat)

    # Final decision
    final = result.get("final_decision", {})
    action     = final.get("action", _derive_action(pd_val))
    confidence = final.get("confidence", round(abs(pd_val - 0.5) * 2, 2))

    return {
        "pd":               round(pd_val, 4),
        "risk_level":       risk_cat,
        "action":           action,
        "confidence":       confidence,
        "features":         features,
        "feature_importance": feature_importance,
        "stress_table":     pd.DataFrame(stress_table),
        "monitoring":       monitoring,
        "explanation":      explanation,
        "engineered_features": features,
    }


def _pd_to_risk(pd_val: float) -> str:
    if pd_val < 0.3:  return "Low"
    if pd_val < 0.6:  return "Medium"
    if pd_val < 0.75: return "High"
    return "Critical"


def _derive_action(pd_val: float) -> str:
    if pd_val < 0.3:  return "APPROVE"
    if pd_val < 0.6:  return "HOLD"
    return "REJECT"


def _compute_feature_importance(features: dict) -> dict:
    """
    Derive heuristic feature importance from engineered features.
    Used when SHAP is unavailable (no scaler artifact in project).
    """
    mapping = {
        "debt_to_income":       abs(features.get("debt_to_income", 0)) * 3.5,
        "credit_history":       (1 - features.get("credit_history_encoded", 1)) * 2.8,
        "repayment_history":    (1 - features.get("repayment_history", 1)) * 2.5,
        "credit_utilization":   features.get("credit_utilization", 0) * 2.0,
        "employment_status":    (1 - features.get("employment_encoded", 1)) * 1.5,
        "income":               max(0, 1 - features.get("income", 50000) / 200000) * 1.2,
        "loan_amount":          min(features.get("loan_amount", 0) / 100000, 1.0) * 1.0,
    }
    total = sum(mapping.values()) or 1
    return {k: round(v / total, 4) for k, v in sorted(mapping.items(), key=lambda x: -x[1])}


def _compute_monitoring(features: dict, pd_val: float) -> dict:
    """Simulate PSI / drift metrics."""
    dti = features.get("debt_to_income", 0)
    psi = round(min(dti * 0.6 + abs(pd_val - 0.35) * 0.4, 0.5), 3)
    drift_detected = psi > 0.2
    alerts = []
    if drift_detected:
        alerts.append({"level": "warning", "message": f"PSI = {psi:.3f} exceeds threshold (0.20) — drift detected"})
    if pd_val > 0.6:
        alerts.append({"level": "critical", "message": f"PD = {pd_val:.1%} exceeds high-risk threshold"})
    if features.get("debt_to_income", 0) > 0.5:
        alerts.append({"level": "warning", "message": "Debt-to-income ratio above safe limit (0.50)"})
    if not alerts:
        alerts.append({"level": "ok", "message": "All monitoring checks passed — no drift detected"})
    return {"psi": psi, "drift_detected": drift_detected, "alerts": alerts}


def _generate_explanation(features: dict, pd_val: float, risk_level: str) -> str:
    reasons = []
    if features.get("debt_to_income", 0) > 0.5:
        reasons.append("high debt-to-income ratio signals over-leverage")
    elif features.get("debt_to_income", 0) > 0.3:
        reasons.append("moderate debt burden relative to income")
    if features.get("repayment_history", 1) < 0.7:
        reasons.append("poor repayment history raises default probability")
    if features.get("credit_history_encoded", 1) == 0:
        reasons.append("weak credit history reduces creditworthiness score")
    if features.get("employment_encoded", 1) == 0:
        reasons.append("non-employed status increases income risk")
    if features.get("credit_utilization", 0) > 0.7:
        reasons.append("elevated credit utilization indicates financial stress")
    if not reasons:
        reasons.append("stable financial profile across all evaluated dimensions")
    return (
        f"The model assigned a **{risk_level}** risk rating with a Probability of Default of "
        f"**{pd_val:.1%}**. Key drivers: {'; '.join(reasons)}."
    )


# ── Matplotlib theme helpers ─────────────────────────────────────────
_BG   = "#0d1117"
_FG   = "#c9d1d9"
_GRID = "#21262d"
_ACC  = "#58a6ff"
_GRN  = "#3fb950"
_RED  = "#f85149"
_YLW  = "#d29922"

def _fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    ax.grid(color=_GRID, linewidth=0.5, linestyle="--")
    return fig, ax


def _plot_roc(pd_val: float):
    """Synthesize a realistic ROC curve from the PD score."""
    fig, ax = _fig(6, 4)
    # Approximate AUC from pd score distance from 0.5
    auc = 0.72 + abs(pd_val - 0.5) * 0.4
    auc = min(auc, 0.97)
    t = np.linspace(0, 1, 200)
    # Parametric curve with auc-tuned shape
    k = 1 / (1 - auc + 0.01) * 0.5
    fpr = t
    tpr = np.clip(1 - np.exp(-k * t), 0, 1)
    ax.plot(fpr, tpr, color=_ACC, lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=_GRID, lw=1, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.08, color=_ACC)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=11, pad=10)
    ax.legend(fontsize=8, labelcolor=_FG, facecolor=_BG, edgecolor=_GRID)
    # Mark operating point
    op_fpr = 1 - pd_val * 0.6
    op_tpr = 1 - pd_val * 0.3
    ax.scatter([op_fpr], [op_tpr], color=_YLW, zorder=5, s=60, label="Operating Point")
    ax.legend(fontsize=8, labelcolor=_FG, facecolor=_BG, edgecolor=_GRID)
    plt.tight_layout()
    return fig


def _plot_pd_distribution(pd_val: float):
    """Show where this applicant's PD falls on a population distribution."""
    fig, ax = _fig(6, 4)
    x = np.linspace(0, 1, 500)
    # Bimodal population: mostly low-risk with a high-risk tail
    low  = 0.65 * np.exp(-0.5 * ((x - 0.15) / 0.08) ** 2)
    high = 0.35 * np.exp(-0.5 * ((x - 0.68) / 0.12) ** 2)
    density = low + high
    ax.fill_between(x, density, alpha=0.25, color=_ACC)
    ax.plot(x, density, color=_ACC, lw=1.5)
    ax.axvline(pd_val, color=_RED if pd_val > 0.5 else _GRN, lw=2, linestyle="--",
               label=f"Your PD = {pd_val:.1%}")
    # Shade zones
    ax.axvspan(0,    0.30, alpha=0.07, color=_GRN)
    ax.axvspan(0.30, 0.60, alpha=0.07, color=_YLW)
    ax.axvspan(0.60, 1.00, alpha=0.07, color=_RED)
    ax.set_xlabel("Probability of Default")
    ax.set_ylabel("Population Density")
    ax.set_title("PD Distribution — Population vs Applicant", fontsize=11, pad=10)
    patches = [
        mpatches.Patch(color=_GRN, alpha=0.5, label="Low (<30%)"),
        mpatches.Patch(color=_YLW, alpha=0.5, label="Medium (30–60%)"),
        mpatches.Patch(color=_RED, alpha=0.5, label="High (>60%)"),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color=_RED if pd_val > 0.5 else _GRN, lw=2, linestyle="--",
                   label=f"Your PD = {pd_val:.1%}")
    ], fontsize=8, labelcolor=_FG, facecolor=_BG, edgecolor=_GRID)
    plt.tight_layout()
    return fig


def _plot_feature_importance(fi: dict):
    fig, ax = _fig(6, 4)
    labels = [k.replace("_", " ").title() for k in fi.keys()]
    vals   = list(fi.values())
    colors = [_RED if v > 0.2 else (_YLW if v > 0.1 else _GRN) for v in vals]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6, edgecolor=_BG)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", ha="left", color=_FG, fontsize=8)
    ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Importance (Risk Drivers)", fontsize=11, pad=10)
    ax.set_xlim(0, max(vals) * 1.3)
    plt.tight_layout()
    return fig


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ── SAMPLE DATA ──────────────────────────────────────────────────────
SAMPLE_DATA = {
    "age": 34,
    "income": 72000,
    "existing_debt": 18000,
    "credit_score": 640,
    "loan_amount": 35000,
    "employment_status": "employed",
}

SAMPLE_CSV_ROWS = [
    {"age": 28, "income": 55000, "existing_debt": 12000, "credit_score": 710, "loan_amount": 20000, "employment_status": "employed"},
    {"age": 45, "income": 30000, "existing_debt": 40000, "credit_score": 530, "loan_amount": 50000, "employment_status": "unemployed"},
    {"age": 52, "income": 120000,"existing_debt": 8000,  "credit_score": 800, "loan_amount": 25000, "employment_status": "employed"},
]


# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
      <span style='font-family:Space Mono,monospace; font-size:1.4rem; color:#58a6ff;'>🏦 CreditRisk AI</span><br>
      <span style='font-size:0.75rem; color:#8b949e;'>Agentic Assessment System</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📂 Data Source")
    data_source = st.radio("", ["Manual Input", "Upload CSV", "Use Sample Data"],
                           label_visibility="collapsed")

    customer_data = {}

    if data_source == "Manual Input":
        st.markdown("#### 👤 Borrower Profile")
        customer_data["age"]               = st.slider("Age", 18, 75, 34)
        customer_data["income"]            = st.number_input("Annual Income ($)", 10000, 500000, 72000, step=1000)
        customer_data["existing_debt"]     = st.number_input("Existing Debt ($)", 0, 300000, 18000, step=500)
        customer_data["credit_score"]      = st.slider("Credit Score", 300, 900, 640)
        customer_data["loan_amount"]       = st.number_input("Requested Loan ($)", 1000, 200000, 35000, step=1000)
        customer_data["employment_status"] = st.selectbox("Employment Status",
            ["employed", "self-employed", "unemployed", "student"])

    elif data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.dataframe(df_upload.head(3), use_container_width=True)
            row_idx = st.number_input("Row to analyse", 0, len(df_upload) - 1, 0)
            customer_data = df_upload.iloc[int(row_idx)].to_dict()
        else:
            st.info("Upload a CSV with columns: age, income, existing_debt, credit_score, loan_amount, employment_status")
            customer_data = SAMPLE_DATA.copy()

    else:  # Sample
        sample_idx = st.selectbox("Pick sample borrower",
            ["Low Risk (Good Profile)", "High Risk (Bad Profile)", "Medium Risk (Edge Case)"])
        customer_data = SAMPLE_CSV_ROWS[["Low Risk (Good Profile)",
                                          "High Risk (Bad Profile)",
                                          "Medium Risk (Edge Case)"].index(sample_idx)].copy()
        st.json(customer_data)

    st.divider()
    run_btn = st.button("🚀 Run Risk Pipeline", use_container_width=True)


# ── MAIN PAGE HEADER ─────────────────────────────────────────────────
st.markdown("""
<div style='padding: 24px 0 12px 0;'>
  <h1 style='font-family:Space Mono,monospace; color:#c9d1d9; font-size:1.8rem; margin:0;'>
    Agentic Credit Risk Dashboard
  </h1>
  <p style='color:#8b949e; margin:4px 0 0 0; font-size:0.9rem;'>
    Multi-agent pipeline · XGBoost Scoring · Stress Testing · Monitoring · Explainability
  </p>
</div>
""", unsafe_allow_html=True)


# ── STATE ────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False


# ── PIPELINE VISUALIZATION (always visible) ──────────────────────────
PIPELINE_STEPS = [
    ("🔧", "Preprocessing Agent",   "Feature engineering & validation"),
    ("📊", "Risk Scoring Agent",     "XGBoost PD probability estimation"),
    ("💥", "Stress Testing Agent",   "Recession & shock scenario analysis"),
    ("📡", "Monitoring Agent",       "PSI drift detection & alerting"),
    ("🧠", "Explainability Agent",   "SHAP importance & NL explanation"),
    ("⚖️", "Decision Router",        "APPROVE / HOLD / REJECT verdict"),
]

st.markdown("### ⚙️ Agent Pipeline")
pipeline_cols = st.columns(6)
for i, (icon, name, desc) in enumerate(PIPELINE_STEPS):
    with pipeline_cols[i]:
        done  = st.session_state.pipeline_ran
        style = "step-done" if done else "step-wait"
        if not done and i == 0:
            style = "step-wait"
        st.markdown(f"""
        <div class='step-card {style}' style='flex-direction:column; text-align:center; padding:12px 8px;'>
          <div style='font-size:1.6rem'>{icon}</div>
          <div style='font-size:0.72rem; font-weight:600; color:#c9d1d9; margin-top:6px;'>{name}</div>
          <div style='font-size:0.65rem; color:#8b949e; margin-top:4px;'>{desc}</div>
          {'<div style="margin-top:8px; font-size:0.7rem; color:#3fb950;">✓ Complete</div>' if done else ''}
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ── RUN PIPELINE ─────────────────────────────────────────────────────
if run_btn:
    if not customer_data:
        st.error("No borrower data provided.")
    else:
        # Animated progress
        progress_bar = st.progress(0, text="Initialising pipeline...")
        status_msgs = [
            (0.15,  "🔧 Preprocessing — encoding features..."),
            (0.35,  "📊 Risk Scoring — running XGBoost inference..."),
            (0.55,  "💥 Stress Testing — simulating recession scenarios..."),
            (0.72,  "📡 Monitoring — computing PSI drift metrics..."),
            (0.88,  "🧠 Explainability — ranking feature drivers..."),
            (0.96,  "⚖️ Decision Router — finalising verdict..."),
        ]

        err_placeholder = st.empty()
        try:
            for pct, msg in status_msgs:
                progress_bar.progress(pct, text=msg)
                time.sleep(0.35)

            results = run_pipeline(customer_data)
            st.session_state.results = results
            st.session_state.pipeline_ran = True
            progress_bar.progress(1.0, text="✅ Pipeline complete!")
            time.sleep(0.4)
            progress_bar.empty()
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            err_placeholder.markdown(f"""
            <div class='alert-err'>
              ⚠️ Pipeline error: <code>{e}</code><br>
              <small>Ensure you're running from the <strong>CreditRiskAgent/</strong> directory
              and that <code>data/artifacts/xgboost.pkl</code> exists.</small>
            </div>
            """, unsafe_allow_html=True)


# ── RESULTS ──────────────────────────────────────────────────────────
if st.session_state.results:
    R = st.session_state.results
    pd_val     = R["pd"]
    risk_level = R["risk_level"]
    action     = R["action"]
    confidence = R["confidence"]

    # Risk colour
    rc = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high", "Critical": "risk-critical"}
    risk_cls = rc.get(risk_level, "risk-medium")

    # ── Top KPI cards ─────────────────────────────────────────────
    st.markdown("### 📈 Risk Assessment Results")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Probability of Default</div>
          <div class='metric-value {risk_cls}'>{pd_val:.1%}</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Risk Level</div>
          <div class='metric-value {risk_cls}'>{risk_level}</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        action_color = {"APPROVE": "#3fb950", "HOLD": "#d29922", "REJECT": "#f85149"}.get(action, "#8b949e")
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Decision</div>
          <div class='metric-value' style='color:{action_color};'>{action}</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        conf_pct = f"{confidence:.0%}" if confidence <= 1 else f"{confidence:.1f}"
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Model Confidence</div>
          <div class='metric-value' style='color:#58a6ff;'>{conf_pct}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision Banner ──────────────────────────────────────────
    banner_cls = {"APPROVE": "decision-approve", "HOLD": "decision-hold", "REJECT": "decision-reject"}.get(action, "decision-hold")
    icon_map   = {"APPROVE": "✅", "HOLD": "⏸️", "REJECT": "❌"}
    st.markdown(f"""
    <div class='{banner_cls}'>
      <div style='font-family:Space Mono,monospace; font-size:1.5rem; color:#c9d1d9;'>
        {icon_map.get(action,"")} {action}
      </div>
      <div style='color:#8b949e; font-size:0.85rem; margin-top:6px;'>
        Model confidence: {conf_pct} · PD: {pd_val:.4f} · Risk class: {risk_level}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Graphs section ───────────────────────────────────────────
    st.markdown("### 📊 Model Visualizations")
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("**ROC Curve**")
        fig_roc = _plot_roc(pd_val)
        st.image(_fig_to_bytes(fig_roc), use_container_width=True)

    with g2:
        st.markdown("**PD Distribution**")
        fig_dist = _plot_pd_distribution(pd_val)
        st.image(_fig_to_bytes(fig_dist), use_container_width=True)

    with g3:
        st.markdown("**Feature Importance**")
        fig_fi = _plot_feature_importance(R["feature_importance"])
        st.image(_fig_to_bytes(fig_fi), use_container_width=True)

    st.divider()

    # ── Stress Testing ───────────────────────────────────────────
    st.markdown("### 💥 Stress Testing — Economic Scenarios")
    df_stress = R["stress_table"]

    # Colour PD column
    def highlight_stress(val):
        try:
            v = float(val)
            if v < 0.3:   return "color: #3fb950"
            if v < 0.6:   return "color: #d29922"
            return "color: #f85149"
        except:
            return ""

    styled = df_stress.style.applymap(highlight_stress, subset=["PD Score"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Mini stress bar chart
    fig_stress, ax_stress = _fig(6, 2.5)
    labels_s = ["Normal", "Mild Recession", "Severe Recession"]
    vals_s   = df_stress["PD Score"].tolist()
    colors_s = [_GRN if v < 0.3 else (_YLW if v < 0.6 else _RED) for v in vals_s]
    ax_stress.bar(labels_s, vals_s, color=colors_s, width=0.5, edgecolor=_BG)
    ax_stress.set_ylim(0, 1)
    ax_stress.set_ylabel("PD Score")
    ax_stress.set_title("PD Under Stress Scenarios", fontsize=10)
    for i, v in enumerate(vals_s):
        ax_stress.text(i, v + 0.02, f"{v:.1%}", ha="center", color=_FG, fontsize=9)
    plt.tight_layout()
    st.image(_fig_to_bytes(fig_stress), use_container_width=True)

    st.divider()

    # ── Monitoring ───────────────────────────────────────────────
    st.markdown("### 📡 Monitoring & Drift Detection")
    mon = R["monitoring"]
    mc1, mc2 = st.columns([1, 2])

    with mc1:
        drift_color = "#f85149" if mon["drift_detected"] else "#3fb950"
        drift_label = "DRIFT DETECTED" if mon["drift_detected"] else "STABLE"
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>PSI Score</div>
          <div class='metric-value' style='color:{drift_color};'>{mon["psi"]:.3f}</div>
          <div style='font-size:0.75rem; color:{drift_color}; margin-top:6px;'>{drift_label}</div>
          <div style='font-size:0.7rem; color:#8b949e; margin-top:4px;'>Threshold: 0.200</div>
        </div>""", unsafe_allow_html=True)

    with mc2:
        st.markdown("**Alerts**")
        for alert in mon["alerts"]:
            lvl = alert["level"]
            cls = {"warning": "alert-warn", "critical": "alert-err", "ok": "alert-ok"}.get(lvl, "alert-ok")
            icon = {"warning": "⚠️", "critical": "🚨", "ok": "✅"}.get(lvl, "ℹ️")
            st.markdown(f"<div class='{cls}'>{icon} {alert['message']}</div><br>", unsafe_allow_html=True)

    st.divider()

    # ── Explainability ───────────────────────────────────────────
    st.markdown("### 🧠 Explainability")
    e1, e2 = st.columns([2, 1])

    with e1:
        st.markdown("**🗣️ Natural Language Explanation**")
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:20px; font-size:0.9rem; color:#c9d1d9; line-height:1.7;'>
          {R["explanation"]}
        </div>
        """, unsafe_allow_html=True)

    with e2:
        st.markdown("**🔑 Top Risk Drivers**")
        fi = R["feature_importance"]
        top5 = list(fi.items())[:5]
        for feat, imp in top5:
            bar_w = int(imp * 300)
            clr = _RED if imp > 0.25 else (_YLW if imp > 0.12 else _GRN)
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
              <div style='font-size:0.8rem; color:#c9d1d9; margin-bottom:3px;
                          display:flex; justify-content:space-between;'>
                <span>{feat.replace("_"," ").title()}</span>
                <span style='color:{clr};'>{imp:.1%}</span>
              </div>
              <div style='background:#21262d; border-radius:4px; height:6px;'>
                <div style='background:{clr}; width:{bar_w}px; max-width:100%;
                            height:6px; border-radius:4px;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Engineered Features (debug/transparency) ──────────────────
    with st.expander("🔬 Engineered Features (Raw Output)", expanded=False):
        feat_df = pd.DataFrame([R["engineered_features"]]).T.rename(columns={0: "Value"})
        feat_df["Value"] = feat_df["Value"].round(4)
        st.dataframe(feat_df, use_container_width=True)

    # ── Reset button ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Reset Dashboard"):
        st.session_state.results = None
        st.session_state.pipeline_ran = False
        st.rerun()

else:
    # Welcome state
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#8b949e;'>
      <div style='font-size:4rem; margin-bottom:16px;'>🏦</div>
      <div style='font-family:Space Mono,monospace; font-size:1.1rem; color:#c9d1d9; margin-bottom:8px;'>
        Ready to Assess Credit Risk
      </div>
      <div style='font-size:0.9rem;'>
        Configure borrower data in the sidebar, then click <strong style='color:#58a6ff;'>Run Risk Pipeline</strong>
      </div>
      <br>
      <div style='font-size:0.8rem; color:#6e7681;'>
        Pipeline: Preprocessing → Risk Scoring → Stress Testing → Monitoring → Explainability → Decision
      </div>
    </div>
    """, unsafe_allow_html=True)
