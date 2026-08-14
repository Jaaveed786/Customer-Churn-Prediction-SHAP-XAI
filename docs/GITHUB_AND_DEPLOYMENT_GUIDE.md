# 🚀 Step-by-Step GitHub Upload & Streamlit Cloud Deployment Guide

This guide walks you through publishing your project to **GitHub** and deploying it for **FREE on Streamlit Community Cloud**.

---

## 📌 Step 1: Upload Project to GitHub

### Option A: Using GitHub Desktop (Easiest — No CLI Needed)
1. Download & open [GitHub Desktop](https://desktop.github.com/).
2. Click **File** → **Add Local Repository...**
3. Choose folder: `E:\Projects for resumes\Customer Churn Prediction with Explainable AI`
4. Click **Publish Repository** → name it `Customer-Churn-Prediction-SHAP-XAI`.
5. Make sure **Keep this code private** is UNCHECKED (make it Public so Streamlit Cloud can access it).
6. Click **Publish Repository**. Done!

---

### Option B: Using VS Code (Integrated GUI)
1. Open VS Code and open folder `E:\Projects for resumes\Customer Churn Prediction with Explainable AI`.
2. Click the **Source Control** icon on the left sidebar (or press `Ctrl + Shift + G`).
3. Click **Publish to GitHub**.
4. Choose **Publish to GitHub public repository**.
5. Select all files and click **OK**. Done!

---

### Option C: Using Git CLI (PowerShell / Command Prompt)
If you have Git installed on your machine, open terminal in project directory:

```powershell
cd "E:\Projects for resumes\Customer Churn Prediction with Explainable AI"

# 1. Initialize git
git init

# 2. Add all files
git add .

# 3. Create initial commit
git commit -m "feat: Initial commit of Customer Churn & Explainable AI (SHAP) Dashboard"

# 4. Link to your GitHub repo (create repo on github.com first)
git remote add origin https://github.com/YOUR_USERNAME/Customer-Churn-Prediction-SHAP-XAI.git

# 5. Push to GitHub
git branch -M main
git push -u origin main
```

---

## 🌐 Step 2: Deploy Free on Streamlit Community Cloud

Once your project is on GitHub, deploy it live in 2 minutes:

1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Fill in the fields:
   * **Repository:** `YOUR_USERNAME/Customer-Churn-Prediction-SHAP-XAI`
   * **Branch:** `main`
   * **Main file path:** `app/streamlit_app.py`
4. Click **Deploy!**

Streamlit Cloud will automatically install `requirements.txt`, launch the dashboard, and give you a live shareable URL (e.g. `https://customer-churn-shap.streamlit.app`) to put at the top of your resume!

---

## 📁 Summary of All Project Document Files

All documentation files are located in `docs/`:

1. [`docs/RESUME_BULLETS.md`](file:///E:/Projects%20for%20resumes/Customer%20Churn%20Prediction%20with%20Explainable%20AI/docs/RESUME_BULLETS.md) — Ready-to-use resume bullet points, skills tags, and metrics tables.
2. [`docs/INTERVIEW_GUIDE.md`](file:///E:/Projects%20for%20resumes/Customer%20Churn%20Prediction%20with%20Explainable%20AI/docs/INTERVIEW_GUIDE.md) — STAR elevator pitch, top 10 Q&A, cost asymmetry math.
3. [`docs/EXPLAINABLE_AI_NOTES.md`](file:///E:/Projects%20for%20resumes/Customer%20Churn%20Prediction%20with%20Explainable%20AI/docs/EXPLAINABLE_AI_NOTES.md) — Game theory Shapley value axioms and SHAP visual explanations.
4. [`docs/BUILT_WITH_ANTIGRAVITY.md`](file:///E:/Projects%20for%20resumes/Customer%20Churn%20Prediction%20with%20Explainable%20AI/docs/BUILT_WITH_ANTIGRAVITY.md) — Antigravity AI engineering workflow doc.
