# 🔮 Customer Churn Prediction with Explainable AI

> **Predict telecom customer churn and explain predictions using SHAP, Streamlit, and scikit-learn.**
> Full ML lifecycle: EDA → Feature Engineering → Model Comparison → SHAP Explainability → Web Dashboard.

---

## 📌 Project Summary

This project predicts customer churn on the IBM Telco dataset using Logistic Regression and Random Forest, compares them across business metrics, and uses SHAP to explain individual predictions. The result is an interactive Streamlit app where stakeholders input customer details and see risk scores plus top predictive drivers.

**Key Result:** Logistic Regression achieves **ROC-AUC = 0.8461** with **90.37% Recall** at our business-justified threshold of **0.35**, while Random Forest achieves **ROC-AUC = 0.8382** with **86.90% Recall**.

---

## 📚 Documentation & Interview Guides

| Guide | Description |
|---|---|
| 📄 [**Resume Bullets & Keywords**](docs/RESUME_BULLETS.md) | Ready-to-use STAR resume bullets, skills tags, and metrics summary tables |
| 🚀 [**GitHub & Deployment Guide**](docs/GITHUB_AND_DEPLOYMENT_GUIDE.md) | Step-by-step instructions to upload to GitHub and deploy for FREE on Streamlit Cloud |
| 📖 [**Interview Guide**](docs/INTERVIEW_GUIDE.md) | Top 10 DS interview questions, STAR method elevator pitch, threshold math, and trade-off narratives |
| 🔬 [**Explainable AI & SHAP Notes**](docs/EXPLAINABLE_AI_NOTES.md) | Deep-dive study notes on Shapley values, game theory axioms, TreeExplainer vs LinearExplainer |
| 🤖 [**Built with Antigravity AI**](docs/BUILT_WITH_ANTIGRAVITY.md) | Engineering methodology, gap resolution, design system architecture by Google DeepMind Antigravity |
| 🎨 [**UI Redesign Walkthrough**](walkthrough.md) | Summary of UI fixes, dark glassmorphism styling, and controlled chart bounding |

---

## 🎯 Business Context & Decision Threshold

Acquiring a new customer costs **5–10× more** than retaining an existing one. We use an operating threshold of **0.35** (lowered from default 0.50) because:
- **Missed Churner (False Negative):** ~$300 lost customer lifetime value.
- **Unnecessary Retention Offer (False Positive):** ~$20 outreach offer cost.
- **15:1 Cost Asymmetry:** Operating at a 0.35 cutoff catches 90%+ of churners.

---

## ⚙️ Feature Engineering

| Feature | Formula | Business Hypothesis |
|---|---|---|
| `tenure_group` | Bin tenure: 0-12, 13-24, 25-48, 49-72 | Non-linear tenure effect |
| `charges_per_month_ratio` | TotalCharges ÷ (tenure+1) | Detects billing anomalies |
| `num_services` | Sum of active add-ons (0-9) | Product engagement proxy |
| `has_security_or_backup` | OnlineSecurity OR OnlineBackup | Sticky service bundling |
| `is_auto_payment` | Payment in {bank transfer, credit card} | Passive payment inertia |
| `contract_risk` | Month-to-month=2, 1yr=1, 2yr=0 | Ordinal contract risk |

---

## 📊 Model Performance Comparison

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **ROC-AUC** | **0.8461** | **0.8382** |
| **5-Fold CV AUC** | **0.8477 ± 0.0122** | **0.8437 ± 0.0096** |
| **Recall (Churn = 1 at 0.35)** | **90.37%** | **86.90%** |
| **Precision (Churn = 1 at 0.35)** | **45.13%** | **47.17%** |
| **F1 Score** | **0.6020** | **0.6115** |

---

## 🚀 Step-by-Step How to Run

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (13 Passing Tests)
```powershell
pytest tests/ -v
```

### 3. Train Models & Save Explainers
```powershell
python -m src.train
```

### 4. Generate SHAP Summary Figures
```powershell
python -m src.generate_shap_plots
```

### 5. Launch Streamlit Web Dashboard
```powershell
python -m streamlit run app/streamlit_app.py
```

---

## 🧪 Test Suite Coverage

```
tests/test_feature_engineering.py  8 passed (feature range, copies, contract risk)
tests/test_preprocessor.py         2 passed (zero data leakage, feature names shape)
tests/test_predict.py              3 passed (end-to-end probability, threshold trigger)
--------------------------------------------------------------------------------------
TOTAL                               13 passed in ~3.4s
```

---

## 🛠️ Tech Stack

`pandas` · `scikit-learn` · `shap` · `streamlit` · `matplotlib` · `seaborn` · `pytest`
