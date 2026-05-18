# Churn Prediction — End-to-End ML Pipeline

A complete, production-style ML project teaching every step from raw data
to a deployed REST API. Built for learning — every file is heavily commented.

---

## Project Structure

```
churn_project/
├── data/
│   ├── raw/                  ← original CSV, never edited
│   └── processed/            ← cleaned, features, predictions
├── src/
│   ├── step1_clean.py        ← STEP 1: Data cleaning
│   ├── step2_eda.py          ← STEP 2: Exploratory analysis
│   ├── step3_features.py     ← STEP 3: Feature engineering
│   ├── step4_train.py        ← STEP 4: Model training
│   ├── step5_predict.py      ← STEP 5: Inference / predict
│   ├── app.py                ← STEP 6: FastAPI deployment
│   └── pipeline.py           ← Orchestrator (runs all steps)
├── models/
│   ├── best_model.pkl        ← saved trained model
│   └── metrics.json          ← model performance metrics
├── tests/
│   └── test_pipeline.py      ← pytest unit tests
├── notebooks/
│   └── eda_plots/            ← saved EDA charts
├── .github/
│   └── workflows/ci.yml      ← GitHub Actions CI/CD
├── Dockerfile
└── requirements.txt
```

---

## Quickstart (local)

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/churn-pipeline.git
cd churn-pipeline
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the full pipeline end-to-end
python src/pipeline.py

# 3. Run tests
pytest tests/ -v

# 4. Start the API
uvicorn src.app:app --reload

# 5. Test the API (in another terminal)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_days": 60, "monthly_spend_usd": 29, "logins_per_month": 1,
       "features_used": 2, "support_tickets_last_90d": 5,
       "days_since_last_login": 60, "nps_score": 3, "team_size": 5,
       "plan": "basic", "acquisition_channel": "paid search",
       "industry": "retail", "support_tier": "none", "is_annual_contract": 0}'
```

---

## Step-by-step learning guide

### Step 1 — Data Cleaning (`src/step1_clean.py`)
- Parse dates, normalise strings
- Remove duplicates
- Impute missing values (median for numeric, "unknown" for categorical)
- Add first derived features (tenure_months, engagement_score)

### Step 2 — EDA (`src/step2_eda.py`)
- Overview stats and churn rate
- Churn rate by every categorical feature
- Distribution plots comparing churned vs retained
- Correlation heatmap

### Step 3 — Feature Engineering (`src/step3_features.py`)
- Compute spend_per_feature, tickets_per_month, recency_ratio
- Ordinal encode plan tier
- One-hot encode categoricals
- Drop PII columns (name, email)

### Step 4 — Model Training (`src/step4_train.py`)
- Split train/test with stratification
- Compare 3 models with 5-fold cross-validation
- Evaluate winner on held-out test set (AUC, precision, recall)
- Save model + metrics to disk

### Step 5 — Predict (`src/step5_predict.py`)
- Load saved model
- Align new data to expected feature columns
- Output churn probability + risk tier

### Step 6 — API (`src/app.py`)
- FastAPI REST API with /predict and /predict/batch endpoints
- Pydantic input validation
- Returns probability, prediction, risk tier

---

## Deploy for free

### Option A — Render (easiest)
1. Push to GitHub
2. Go to render.com → New Web Service → connect repo
3. Set start command: `uvicorn src.app:app --host 0.0.0.0 --port 8000`
4. Done — free tier available

### Option B — Railway
1. Push to GitHub
2. railway.app → New Project → Deploy from GitHub
3. Add env var `PORT=8000`
4. Free $5 credits/month

### Option C — Docker anywhere
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## CI/CD (GitHub Actions)

On every push:
1. **Lint** — ruff checks code style
2. **Test** — pytest runs unit tests
3. **Train** — full pipeline runs, model is saved
4. **Docker** — image is built and smoke-tested

Free tier: 2,000 minutes/month on GitHub.

---

## Key concepts learned

| Concept | Where |
|---|---|
| Missing value imputation | step1_clean.py |
| Exploratory analysis | step2_eda.py |
| Feature engineering | step3_features.py |
| Model selection & evaluation | step4_train.py |
| Model serialisation | step4_train.py (joblib) |
| REST API design | src/app.py |
| Containerisation | Dockerfile |
| CI/CD pipeline | .github/workflows/ci.yml |
| Unit testing | tests/test_pipeline.py |
