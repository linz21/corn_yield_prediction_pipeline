# 🌽 Corn Yield Prediction — Full MLOps Pipeline

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A production-grade ML system that predicts US corn yield (bu/acre) with **Bayesian bootstrap confidence intervals**, full experiment tracking, drift monitoring, and CI/CD deployment.

## Architecture

```
USDA NASS API → Data Pipeline (DVC) → XGBoost + TabNet
                                            ↓
                                    MLflow Experiment Tracking
                                            ↓
                               FastAPI /predict (with 95% CI)
                                            ↓
                            Docker → GitHub Actions → AWS EC2
                                            ↓
                               Evidently AI Drift Monitor
```

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/linz21/corn-yield-prediction
cd corn-yield-prediction
pip install -r requirements.txt

# 2. Get data (demo — no API key needed)
python src/data/ingest.py --demo

# 3. Train model
python src/models/train.py

# 4. View experiments
mlflow ui   # → http://localhost:5000

# 5. Serve API
uvicorn src.api.main:app --reload   # → http://localhost:8000/docs
```

## Sample prediction with confidence interval

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "state": "Illinois", "precip_inches": 28.5, "temp_avg_f": 58.2}'
```

```json
{
  "predicted_yield_bu_per_acre": 198.4,
  "ci_lower": 183.1,
  "ci_upper": 213.7,
  "confidence_level": 0.95,
  "ci_width": 30.6,
  "latency_ms": 42.1
}
```

## Tech Stack

`XGBoost` · `MLflow` · `DVC` · `FastAPI` · `Pydantic` · `Evidently AI` · `Docker` · `GitHub Actions` · `AWS EC2`

## Results

| Metric | Value |
|--------|-------|
| Test RMSE | TBD |
| Test R² | TBD |
| CI Coverage (95%) | TBD |
| API Latency (p95) | TBD |
