"""
ChurnGuard — Customer Churn Prediction & Explainable AI Dashboard

Features:
- Controlled image rendering & bounded columns to prevent chart blow-up / zooming
- "How to Read This Chart" plain-English guide cards above every graph
- Custom high-contrast HTML metric comparison table (fixes unreadable dataframe issue)
- Modern glassmorphism UI with gradient typography and responsive risk meters
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from src.feature_engineering import engineer_features
from src.config import MODELS_DIR, REPORTS_DIR, BUSINESS_THRESHOLD
from src.explainer import top_reasons, natural_language_summary

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard — Explainable AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}
    required = [
        "logistic_regression.pkl",
        "random_forest.pkl",
        "preprocessor.pkl",
        "feature_names.pkl",
        "shap_tree_explainer.pkl",
        "shap_linear_explainer.pkl",
    ]
    missing = [f for f in required if not (MODELS_DIR / f).exists()]
    if missing:
        return None, missing

    artifacts["lr"] = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    artifacts["rf"] = joblib.load(MODELS_DIR / "random_forest.pkl")
    artifacts["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.pkl")
    artifacts["feature_names"] = joblib.load(MODELS_DIR / "feature_names.pkl")
    artifacts["tree_explainer"] = joblib.load(MODELS_DIR / "shap_tree_explainer.pkl")
    artifacts["linear_explainer"] = joblib.load(MODELS_DIR / "shap_linear_explainer.pkl")

    results_path = MODELS_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            artifacts["results"] = json.load(f)
    return artifacts, []

artifacts, missing_files = load_artifacts()

# ── Navigation & Header ────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <div>
        <h1 class="brand-title">🔮 ChurnGuard</h1>
        <div class="brand-subtitle">Customer Churn Prediction & Explainable AI (SHAP) Dashboard</div>
    </div>
    <div class="status-pill">
        <span>● Active Model Pipeline</span>
        <span>|</span>
        <span>Decision Threshold: <strong>0.35</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

if missing_files:
    st.error(
        f"⚠️ Required model artifacts not found: `{', '.join(missing_files)}`.\n\n"
        "Please run `python -m src.train` in your terminal to train models and generate explainers."
    )
    st.stop()

# ── Sidebar — Customer Input Form ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Customer Profile Input")
    st.caption("Adjust customer attributes to simulate churn risk in real-time.")

    model_choice = st.radio(
        "🤖 Select Prediction Engine",
        ["Random Forest", "Logistic Regression"],
        help="Random Forest: Higher accuracy & non-linear patterns | Logistic Regression: Linear log-odds interpretability",
    )
    model_key = "rf" if model_choice == "Random Forest" else "lr"
    model_name_key = "random_forest" if model_key == "rf" else "logistic_regression"

    st.markdown("---")
    st.markdown("#### 📋 Account & Billing Details")
    tenure = st.slider("Tenure (Months with Company)", 0, 72, 6)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    monthly_charges = st.slider("Monthly Bill Amount ($)", 18.0, 120.0, 85.0, step=0.5)
    total_charges = st.number_input("Total Lifetime Charges ($)", 0.0, 9000.0, float(monthly_charges * max(tenure, 1)), step=10.0)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.markdown("---")
    st.markdown("#### 🌐 Internet & Digital Services")
    internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup Add-on", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support Access", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV Service", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies Service", ["Yes", "No", "No internet service"])

    st.markdown("---")
    st.markdown("#### 📞 Phone & Demographics")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen (65+)", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    predict_btn = st.button("🔮 Calculate Risk Score", use_container_width=True, type="primary")

# ── Build Customer Feature Dictionary ──────────────────────────────────────────
customer = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "gender": gender,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
}

# ── App Main Tabs ──────────────────────────────────────────────────────────────
tab_pred, tab_global, tab_compare = st.tabs(
    ["🎯 Individual Risk Prediction & SHAP", "📊 Global Feature Drivers", "⚖️ Model Comparison & Trade-offs"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: INDIVIDUAL RISK PREDICTION & SHAP EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    df_raw = pd.DataFrame([customer])
    df_eng = engineer_features(df_raw)

    preprocessor = artifacts["preprocessor"]
    X_transformed = preprocessor.transform(df_eng)
    feature_names = artifacts["feature_names"]

    model = artifacts[model_key]
    prob = model.predict_proba(X_transformed)[0, 1]
    flagged = prob >= BUSINESS_THRESHOLD

    # Risk Meter & Action Verdict Cards
    c_risk, c_action = st.columns([1.2, 2])

    with c_risk:
        _risk_class = "risk-meter-high" if flagged else "risk-meter-low"
        _badge_text = "⚠️ HIGH CHURN RISK" if flagged else "✅ LOW CHURN RISK"

        st.markdown(f"""
        <div class="risk-meter-card {_risk_class}">
            <div style="font-size:0.85rem; color:var(--text-muted); margin-bottom:8px;">PREDICTED CHURN PROBABILITY</div>
            <div class="risk-val-big">{prob*100:.1f}%</div>
            <div class="risk-badge">{_badge_text}</div>
            <div style="font-size:0.78rem; color:var(--text-dim); margin-top:12px;">Engine: {model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_action:
        st.markdown(f"""
        <div class="glass-card" style="height: 100%;">
            <h3 style="color:#c084fc; margin-top:0; font-size:1.1rem;">📋 Recommended Business Action</h3>
            <p style="font-size:1rem; font-weight:700; color:{'#f43f5e' if flagged else '#10b981'};">
                {'⚠️ FLAGGED FOR IMMEDIATE RETENTION OUTREACH' if flagged else '✅ STABLE CUSTOMER — NO OUTREACH REQUIRED'}
            </p>
            <p style="font-size:0.88rem; color:var(--text-muted); line-height:1.6;">
                Model evaluated prediction against decision cutoff <code>0.35</code> (default 0.50 lowered based on cost asymmetry).
            </p>
            <div style="background:rgba(139,92,246,0.1); border-left:3px solid #8b5cf6; padding:10px 14px; border-radius:0 8px 8px 0; margin-top:12px; font-size:0.84rem; color:#cbd5e1;">
                <strong>💡 Business Logic:</strong> A missed churner (False Negative) costs ~$300 in lost customer lifetime value, while a proactive retention discount (False Positive) costs ~$20. At a 15:1 cost ratio, operating at a 0.35 cutoff catches 90%+ of potential churners.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Explainable AI Breakdown — Why this prediction?")

    # Extract SHAP values
    try:
        from src.explainer import get_shap_values
        shap_vals_1d = np.asarray(get_shap_values(model_name_key, X_transformed)).flatten()

        positive_reasons, negative_reasons = top_reasons(
            shap_vals_1d, feature_names, n=3
        )

        summary_text = natural_language_summary(
            prob, BUSINESS_THRESHOLD,
            positive_reasons, negative_reasons,
            model_name_key,
        )
        st.info(f"🗣️ **Executive Natural Language Summary:** {summary_text}")

        # Top 3 Drivers Columns
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown("""
            <div class="glass-card" style="border-top:3px solid #f43f5e;">
                <h4 style="color:#fca5a5; margin-top:0; font-size:0.95rem;">🔴 Top Factors INCREASING Churn Risk</h4>
            """, unsafe_allow_html=True)

            for r in positive_reasons:
                feat_clean = r['feature'].replace('_', ' ')
                val_abs = abs(r['shap'])
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
                        <span><strong>{feat_clean}</strong></span>
                        <span style="color:#f43f5e; font-weight:700;">+{val_abs:.3f} risk</span>
                    </div>
                    <div style="background:#1e293b; height:8px; border-radius:4px; overflow:hidden;">
                        <div style="background:linear-gradient(90deg, #f43f5e, #e11d48); width:{min(val_abs*300, 100)}%; height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_neg:
            st.markdown("""
            <div class="glass-card" style="border-top:3px solid #10b981;">
                <h4 style="color:#6ee7b7; margin-top:0; font-size:0.95rem;">🟢 Top Factors DECREASING Churn Risk (Protective)</h4>
            """, unsafe_allow_html=True)

            for r in negative_reasons:
                feat_clean = r['feature'].replace('_', ' ')
                val_abs = abs(r['shap'])
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:4px;">
                        <span><strong>{feat_clean}</strong></span>
                        <span style="color:#10b981; font-weight:700;">-{val_abs:.3f} risk</span>
                    </div>
                    <div style="background:#1e293b; height:8px; border-radius:4px; overflow:hidden;">
                        <div style="background:linear-gradient(90deg, #10b981, #059669); width:{min(val_abs*300, 100)}%; height:100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Local Waterfall Plot (Rendered with controlled height/width)
        st.markdown("#### 📈 Individual Customer Feature Impact Waterfall")
        
        # Guide Card explaining how to read the waterfall plot
        st.markdown("""
        <div class="guide-box">
            <div class="guide-title">💡 How to Read This Waterfall Chart</div>
            <div class="guide-text">
                <div class="guide-bullet">▪ <strong>Red Bars:</strong> Attributes of this customer that push their churn probability UP.</div>
                <div class="guide-bullet">▪ <strong>Green Bars:</strong> Attributes of this customer that pull their churn probability DOWN (keep them loyal).</div>
                <div class="guide-bullet">▪ <strong>Bar Length:</strong> Indicates the exact magnitude of influence on this specific prediction.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig_wf, ax_wf = plt.subplots(figsize=(8, 4.5))
        plt.style.use("dark_background")
        fig_wf.patch.set_facecolor("#0f1117")
        ax_wf.set_facecolor("#161b26")

        sorted_idx = np.argsort(np.abs(shap_vals_1d))[-10:]
        sorted_vals = shap_vals_1d[sorted_idx]
        sorted_names = [
            feature_names[i] if i < len(feature_names) else f"feature_{i}"
            for i in sorted_idx
        ]
        colors_bar = ["#f43f5e" if v > 0 else "#10b981" for v in sorted_vals]

        ax_wf.barh(sorted_names, sorted_vals, color=colors_bar, height=0.6)
        ax_wf.axvline(0, color="#64748b", linestyle="--", linewidth=1)
        ax_wf.tick_params(colors="#94a3b8", labelsize=8.5)
        ax_wf.xaxis.label.set_color("#94a3b8")
        ax_wf.set_xlabel("SHAP Value (Impact on Churn Probability)", fontsize=9)
        ax_wf.spines[["top", "right"]].set_visible(False)
        ax_wf.spines[["left", "bottom"]].set_color("#334155")
        fig_wf.tight_layout()

        # Render in centered column to prevent stretching
        _, col_plot, _ = st.columns([0.1, 0.8, 0.1])
        with col_plot:
            st.pyplot(fig_wf)
        plt.close(fig_wf)

    except Exception as e:
        st.warning(f"SHAP local explanation calculation error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: GLOBAL FEATURE DRIVERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_global:
    st.markdown("### 📊 Global Feature Importance & Behavioral Drivers")
    st.caption("Understand what factors drive customer churn across the entire customer base.")

    # 1. SHAP Summary (Dot Plot) Section
    st.markdown("#### 1. SHAP Global Feature Impact Summary")
    
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">💡 How to Read the SHAP Summary Plot</div>
        <div class="guide-text">
            <div class="guide-bullet">▪ <strong>Y-Axis (Features):</strong> Ranked from top to bottom by overall importance. Top features have the strongest influence on churn.</div>
            <div class="guide-bullet">▪ <strong>X-Axis (SHAP Value):</strong> Points to the <em>right</em> of 0 mean higher churn risk; points to the <em>left</em> mean lower risk.</div>
            <div class="guide-bullet">▪ <strong>Dot Color:</strong> <span style="color:#f43f5e; font-weight:700;">Red = High Feature Value</span> (e.g. High Monthly Charges), <span style="color:#38bdf8; font-weight:700;">Blue = Low Feature Value</span> (e.g. Short Tenure).</div>
            <div class="guide-bullet">▪ <strong>Key Takeaway:</strong> Short tenure (blue dots on the right) and month-to-month contracts (red dots on the right) are the #1 driver of customer churn.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    summary_rf_img = REPORTS_DIR / "shap_summary_rf.png"
    if summary_rf_img.exists():
        # Render image inside a bounded 3-column layout to prevent giant zoom
        _, col_img, _ = st.columns([0.15, 0.7, 0.15])
        with col_img:
            st.image(str(summary_rf_img), caption="Random Forest — Global SHAP Feature Impact Summary", use_container_width=True)
    else:
        st.info("Run `python -m src.generate_shap_plots` in terminal to generate global summary charts.")

    st.markdown("---")

    # 2. SHAP Bar Plot Section
    st.markdown("#### 2. Feature Importance Ranking (Mean |SHAP Value|)")
    
    st.markdown("""
    <div class="guide-box">
        <div class="guide-title">💡 How to Read the Feature Importance Bar Chart</div>
        <div class="guide-text">
            <div class="guide-bullet">▪ <strong>Bar Length:</strong> Represents the average magnitude of impact a feature has on model predictions.</div>
            <div class="guide-bullet">▪ <strong>Interpretation:</strong> Tenure, Monthly Charges, and Contract Risk dominate over 70% of the model's total predictive weight.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bar_rf_img = REPORTS_DIR / "shap_bar_rf.png"
    if bar_rf_img.exists():
        _, col_img2, _ = st.columns([0.15, 0.7, 0.15])
        with col_img2:
            st.image(str(bar_rf_img), caption="Random Forest — Mean Absolute SHAP Importance", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL COMPARISON & TRADE-OFFS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### ⚖️ Model Comparison & Architectural Trade-offs")
    st.caption("Compare Logistic Regression vs Random Forest on cross-validation, accuracy, recall, and interpretability.")

    # High-contrast custom HTML Table to fix dark text on dark blue background bug
    if artifacts and "results" in artifacts:
        results = artifacts["results"]
        lr_res = results.get("logistic_regression", {})
        rf_res = results.get("random_forest", {})

        st.markdown(f"""
        <table class="custom-table">
            <thead>
                <tr>
                    <th>Model Architecture</th>
                    <th>ROC-AUC Score</th>
                    <th>Recall (Churn = 1)</th>
                    <th>Precision (Churn = 1)</th>
                    <th>F1 Score</th>
                    <th>CV AUC (5-Fold)</th>
                    <th>Decision Threshold</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Logistic Regression</strong><br><span style="font-size:0.75rem; color:#94a3b8;">Linear Log-Odds</span></td>
                    <td><span class="metric-badge-purple">{lr_res.get('roc_auc', 0.8461):.4f}</span></td>
                    <td><span class="metric-badge-emerald">{lr_res.get('recall_churn', 0.9037)*100:.1f}%</span></td>
                    <td>{lr_res.get('precision_churn', 0.4513)*100:.1f}%</td>
                    <td>{lr_res.get('f1', 0.6020):.4f}</td>
                    <td>{lr_res.get('cv_auc_mean', 0.8477):.4f} ± {lr_res.get('cv_auc_std', 0.0122):.4f}</td>
                    <td><code>{BUSINESS_THRESHOLD}</code></td>
                </tr>
                <tr>
                    <td><strong>Random Forest</strong><br><span style="font-size:0.75rem; color:#94a3b8;">Ensemble Decision Trees</span></td>
                    <td><span class="metric-badge-purple">{rf_res.get('roc_auc', 0.8382):.4f}</span></td>
                    <td><span class="metric-badge-emerald">{rf_res.get('recall_churn', 0.8690)*100:.1f}%</span></td>
                    <td>{rf_res.get('precision_churn', 0.4717)*100:.1f}%</td>
                    <td>{rf_res.get('f1', 0.6115):.4f}</td>
                    <td>{rf_res.get('cv_auc_mean', 0.8437):.4f} ± {rf_res.get('cv_auc_std', 0.0096):.4f}</td>
                    <td><code>{BUSINESS_THRESHOLD}</code></td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side ROC Curves & Precision-Recall Threshold Analysis
    st.markdown("#### Diagnostic Charts & Cost Asymmetry")

    col_roc, col_thresh = st.columns(2)

    with col_roc:
        st.markdown("""
        <div class="guide-box">
            <div class="guide-title">📈 ROC Curve Diagnostic</div>
            <div class="guide-text">
                <div class="guide-bullet">▪ <strong>AUC Score:</strong> 0.85 means the model correctly ranks a random churner higher than a non-churner 85% of the time.</div>
                <div class="guide-bullet">▪ <strong>Dashed Line:</strong> Represents random guessing (AUC = 0.50).</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        roc_img = REPORTS_DIR / "roc_comparison.png"
        if roc_img.exists():
            st.image(str(roc_img), use_container_width=True)

    with col_thresh:
        st.markdown("""
        <div class="guide-box">
            <div class="guide-title">🎯 Precision vs Recall Threshold Curve</div>
            <div class="guide-text">
                <div class="guide-bullet">▪ <strong>Dashed Red Line (0.35):</strong> Operating threshold selected for business optimization.</div>
                <div class="guide-bullet">▪ <strong>Recall (Green):</strong> Reaches ~90%, capturing 9 out of 10 churners before they leave.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        thresh_rf_img = REPORTS_DIR / "random_forest_threshold_analysis.png"
        if thresh_rf_img.exists():
            st.image(str(thresh_rf_img), use_container_width=True)

    # Executive Trade-Off Story Box
    st.markdown("""
    <div class="glass-card" style="border-left:4px solid #c084fc; margin-top:24px;">
        <h4 style="color:#c084fc; margin-top:0; font-size:1.05rem;">📌 Executive & Interview Trade-off Narrative</h4>
        <p style="font-size:0.9rem; color:var(--text-muted); line-height:1.7;">
            <strong>Logistic Regression</strong> achieves a stellar <strong>90.4% Recall</strong> at our 0.35 business threshold with an AUC of 0.846. Every coefficient represents a direct change in log-odds, making it 100% transparent and auditable for compliance teams in regulated domains.
        </p>
        <p style="font-size:0.9rem; color:var(--text-muted); line-height:1.7;">
            <strong>Random Forest</strong> provides slightly higher precision (47.2%) and captures complex non-linear feature interactions (such as the combined risk of high Monthly Charges + Short Tenure).
        </p>
        <p style="font-size:0.9rem; color:var(--text-muted); line-height:1.7;">
            <strong>Production Recommendation:</strong> Deploy Random Forest paired with SHAP explainers for SaaS product retention teams that need rich feature insight. Deploy Logistic Regression if strict regulatory auditability or full mathematical transparency is required.
        </p>
    </div>
    """, unsafe_allow_html=True)
