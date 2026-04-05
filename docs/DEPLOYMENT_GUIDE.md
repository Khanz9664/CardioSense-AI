# Streamlit Deployment Guide: CardioSense AI

This guide provides step-by-step instructions for deploying the **CardioSense AI Clinical Dashboard** to **Streamlit Community Cloud**.

---

##  1. Repository Preparation

Before deploying, ensure your GitHub repository contains all the necessary artifacts for the clinical engine to function.

### A. Core File Checklist
Ensure the following are pushed to your repository:
- `app/main.py` (The entry point)
- `src/` (Core logic)
- `models/` (Contains `heart_disease_model.joblib`, `preprocessor.joblib`, and `model_metadata.json`)
- `requirements.txt` (List of dependencies)

### B. Verify Model Paths
The dashboard uses relative paths to load the clinical artifacts. If you encounter a `FileNotFoundError`, verify that your `models/` directory is in the root of the repository.

---

##  2. Deploying to Streamlit Community Cloud

### Step 1: Push to GitHub
If you haven't already, push your code to a public GitHub repository.

```bash
git add .
git commit -m "message"
git push origin main
```

### Step 2: Access Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Log in with your GitHub account.

### Step 3: Deploy New App
1.  Click **"New app"**.
2.  Select your **Repository** (`khanz9664/CardioSense-AI`).
3.  Select the **Branch** (`main`).
4.  **Main file path**: Set this to **`app/main.py`**.
5.  Click **"Deploy!"**.

---

##  3. Advanced Configuration (Optional)

### Handling Secrets
If you decide to add API keys (e.g., for external hospital databases), use the **Streamlit Secrets** manager:
1.  In the Streamlit Cloud dashboard, go to your app's **Settings**.
2.  Select **Secrets**.
3.  Add your keys in TOML format: `API_KEY = "your_secret_here"`.

### Python Version
Streamlit Cloud typically defaults to the latest stable Python version (3.11 or 3.12). Ensure your `requirements.txt` is updated to avoid version conflicts with packages like `xgboost` or `shap`.

---

##  4. Troubleshooting Clinical Artifacts

-   **Memory Errors**: If the app crashes on launch, it may be due to `shap` calculating global importance on a large reference dataset. Ensure `models/model_metadata.json` has pre-calculated features.
-   **Model Hashing**: The "Audit Hash" in the UI will verify that the model in the cloud matches your local validated version.

---

**Your Clinical Decision Support System is now ready for global clinical access!**
