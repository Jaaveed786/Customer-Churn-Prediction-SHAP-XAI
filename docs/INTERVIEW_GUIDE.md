# 🎯 Customer Churn Prediction — Comprehensive DS Interview Guide

> **How to present this project in Data Science & Machine Learning Interviews to stand out and sound like a Senior ML Engineer.**

---

## 🗣️ The 2-Minute Elevator Pitch

*"In this project, I built an end-to-end customer churn prediction engine for a telecom SaaS dataset with Explainable AI at its core.*

*Rather than focusing solely on raw model accuracy, I designed the project around real-world business trade-offs. I engineered 6 domain-specific features—such as contract risk indices and engagement ratios—and trained both Logistic Regression and Random Forest models using 5-fold cross-validation.*

*Because a missed churner costs ~$300 in lost customer lifetime value while a proactive retention offer costs only ~$20, I tuned the classification threshold down to **0.35**, achieving over **90.4% Recall** for high-risk customers.*

*Finally, to make the model trustworthy for non-technical stakeholders, I packaged it into an interactive Streamlit app backed by **SHAP (SHapley Additive exPlanations)**. Every prediction displays the top 3 drivers pushing risk up or down, complete with plain-English executive summaries."*

---

## ❓ Top 10 DS Interview Questions & Winning Answers

### Q1. "Tell me about a classification project you worked on."
**Answer Strategy (STAR Method):**
- **Situation:** Churn reduces recurring revenue in SaaS/telecom businesses; reactive retention is expensive.
- **Task:** Build a model that flags churners 30 days in advance with explainable reasons for outreach teams.
- **Action:**
  1. Cleaned IBM Telco dataset (7,043 rows) and engineered 6 features (`tenure_group`, `charges_per_month_ratio`, `num_services`, etc.).
  2. Built an un-leaked `ColumnTransformer` pipeline fit **only** on training data.
  3. Trained and compared Logistic Regression vs Random Forest.
  4. Lowered decision cutoff to **0.35** based on a 15:1 cost ratio ($300 FN vs $20 FP).
  5. Integrated `SHAP` (`TreeExplainer` & `LinearExplainer`) for local and global model interpretability.
- **Result:** Logistic Regression reached 0.8461 ROC-AUC and 90.37% Recall; Random Forest reached 0.8382 ROC-AUC. Packaged into a Streamlit dashboard.

---

### Q2. "Why compare Logistic Regression and Random Forest? Why not just use XGBoost?"
**Answer:**
*"The goal of a Senior ML Engineer isn't always to ship the highest complex black-box model, but to present honest architectural trade-offs to business stakeholders:*

1. **Logistic Regression** provides exact mathematical transparency. Each coefficient tells us the direct change in log-odds of churn per unit change in a feature. In regulated domains (banking, healthcare, compliance), this auditability is mandatory.
2. **Random Forest** captures non-linear feature interactions (e.g., high Monthly Charges combined with short Tenure) without needing manual interaction terms.

By comparing both, I demonstrated that Logistic Regression gave 90.4% Recall vs Random Forest's 86.9%, while Random Forest gave slightly higher precision (47.2% vs 45.1%). XGBoost could be added in v2, but LR vs RF establishes the core interpretability vs. complexity spectrum."*

---

### Q3. "How did you pick your classification decision threshold? Why not the default 0.50?"
**Answer:**
*"The default 0.50 cutoff assumes equal costs for False Positives and False Negatives. In customer churn, there is a massive **cost asymmetry**:

$$\text{Cost}(\text{False Negative}) \approx \$300 \quad \text{(Lost Customer Lifetime Value)}$$
$$\text{Cost}(\text{False Positive}) \approx \$20 \quad \text{(Cost of a discount/outreach offer)}$$

At a 15:1 cost ratio, a False Negative is 15× more damaging than a False Positive. Operating at a **0.35 threshold** maximizes Recall (catching 90%+ of churners) while keeping False Positives at an acceptable level for the marketing budget."*

---

### Q4. "How did you prevent Data Leakage during preprocessing?"
**Answer:**
*"Data leakage occurs when information from the test dataset leaks into model training. I prevented it using scikit-learn `Pipeline` and `ColumnTransformer`:

1. **Split First:** The dataset was split into 80% train / 20% test *before* any feature scaling or imputation.
2. **Fit on Train Only:** `preprocessor.fit(X_train)` calculated median imputations and standard scaler means/stds strictly on training rows.
3. **Transform Test Separately:** `preprocessor.transform(X_test)` applied those pre-calculated statistics to test rows.

*Note on EDA:* Exploratory analysis was conducted on full data to observe general distributions, but no preprocessing statistics were derived from the full dataset."*

---

### Q5. "What is SHAP, and why use it over Gini Feature Importance?"
**Answer:**
*"Gini feature importance (or mean decrease in impurity) in tree models has two major flaws:
1. It is purely **global**—it cannot tell you why a *specific individual* customer is predicted to churn.
2. It suffers from **bias toward high-cardinality features** and can be misleading when features are correlated.

**SHAP (SHapley Additive exPlanations)** is based on Shapley values from cooperative game theory. It guarantees:
- **Local Explanations:** Breaks down a specific customer's risk into `Base Rate + sum(SHAP values)`.
- **Consistency & Fairness:** Features that truly contribute more always receive larger SHAP magnitude.
- **Directionality:** Explicitly shows whether an attribute *increases* (positive SHAP) or *decreases* (negative SHAP) churn probability."*

---

### Q6. "How did you prevent Training-Serving Skew between your ML pipeline and Streamlit app?"
**Answer:**
*"Training-serving skew occurs when feature logic written during model training differs from inference code in the production app.

To eliminate this risk, I created a single module `src/feature_engineering.py` containing `engineer_features()`.
Both `src/train.py` and `app/streamlit_app.py` import and execute the **exact same function**. The app never reimplements feature calculations inline."*

---

### Q7. "How did you persist SHAP explainers for production inference?"
**Answer:**
*"SHAP `LinearExplainer` requires a background reference dataset to compute marginal expectations.

During model training (`src/train.py`), I fitted both `TreeExplainer(rf_model)` and `LinearExplainer(lr_model, X_train_transformed)` on training data, then serialized the fitted explainer objects to disk as `.pkl` files.

At inference time in Streamlit, the app simply loads `shap_tree_explainer.pkl` or `shap_linear_explainer.pkl`. The background reference data is baked into the serialized object, eliminating the need to re-fit or reload training data at inference time."*

---

### Q8. "What engineered features added the most predictive value?"
**Answer:**
1. `contract_risk`: Ordinal mapping (Month-to-month=2, 1 Year=1, 2 Year=0). Month-to-month customers had 3× higher churn rates.
2. `charges_per_month_ratio`: `TotalCharges / (tenure + 1)`. Uncovered billing anomalies where total charges diverged from expected monthly rates.
3. `num_services`: Sum of active subscribed add-ons (0-9). Higher service adoption correlates strongly with customer retention (sticky product).
4. `is_auto_payment`: Binary indicator for automatic bank/credit card payments. Passive payment inertia reduces active cancellation opportunities.

---

## 📊 Summary of Model Metrics for Reference

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| **ROC-AUC** | **0.8461** | **0.8382** |
| **5-Fold CV AUC** | **0.8477 ± 0.0122** | **0.8437 ± 0.0096** |
| **Recall (at 0.35 threshold)** | **90.37%** | **86.90%** |
| **Precision (at 0.35 threshold)** | **45.13%** | **47.17%** |
| **F1 Score** | **0.6020** | **0.6115** |
