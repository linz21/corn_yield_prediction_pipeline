# 🌽 Corn Yield Prediction — Full MLOps Pipeline

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A production-grade ML system that predicts US corn yield (bu/acre) with **Bootstrap prediction intervals**, full experiment tracking, drift monitoring, and CI/CD deployment.

## 🔗 Live Demo
Deployed on AWS EC2 with a permanent Elastic IP: **http://54.214.151.133:8000/docs**

![API Demo](docs/images/api_demo_screenshot.png)

Try a prediction:
```bash
curl -X POST http://54.214.151.133:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "state": "Illinois", "planted_acres": 61300, "yield_bu_per_acre_lag1": 195, "yield_3yr_avg": 198}'
```

> Note: this demo instance runs intermittently to manage cloud costs. If the link isn't responding, see [Quickstart](#quickstart) to run it locally.


## Architecture

```
USDA NASS API → Data Pipeline (DVC -> S3) → XGBoost
                                            ↓
                                    MLflow Experiment Tracking
                                            ↓
                              Portable Model Export (joblib)
                                            ↓
                               FastAPI /predict (with 95% prediction interval)
                                            ↓
                            Docker → GitHub Actions → AWS EC2
                                            ↓
                          Evidently AI Drift Monitoring (validated ✓)
```
## Quickstart
```bash
# 1. Clone and install
git clone https://github.com/linz21/corn_yield_prediction_pipeline.git
cd corn_yield_prediction_pipeline
pip install -r requirements.txt

# 2. (Optional) Pull DVC-tracked processed data from S3
# Requires AWS credentials configured — see: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html
aws configure
dvc pull

# 3. Get data (demo — no API key needed)
python src/data/ingest.py --demo

# 4. Build features (--demo keeps this isolated from any real production data)
python src/features/build_features.py --demo

# 5. Train model (--config points at the demo data path; exports models/latest_model_demo.pkl)
python src/models/train.py --no-bootstrap --config configs/config_demo.yaml

# 6. View experiments
MLFLOW_ALLOW_FILE_STORE=true mlflow ui   # → http://localhost:5000

# 7. Serve API
uvicorn src.api.main:app --reload   # → http://localhost:8000/docs
```
> **Note:** The steps above use synthetic demo data (240 rows) and are fully isolated from real
> production data/models via `--demo`/`--config configs/config_demo.yaml` — safe to run repeatedly
> without touching anything real.
>
> For the real production pipeline, get a free API key at
> [quickstats.nass.usda.gov/api](https://quickstats.nass.usda.gov/api) and run:
> ```bash
> python src/data/ingest.py --api-key YOUR_USDA_KEY
> python src/features/build_features.py
> python src/models/train.py --run-name "production-model"
> ```
> This trains on the full, real dataset (30,416 rows, after excluding the current in-progress
> season) and exports `models/latest_model.pkl`. Reported honest metrics — R²=0.845, MAPE=6.59%,
> RMSE=13.65 bu/acre — come from a separate evaluation-only model trained on 2000-2014 data alone
> and tested on a genuine chronological 2015-2025 holdout (see "Data exploration & model selection"
> below for the full methodology). The actual deployed model is retrained on all available data
> once that approach is validated — see "Note on the deployed model vs. reported metrics."

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

`XGBoost` · `MLflow` · `DVC` · `FastAPI` · `Pydantic` · `Docker` · `GitHub Actions` · `AWS EC2`

## Results

| Metric                       | Value                                              |
| ----------------------------- | --------------------------------------------------- |
| Test RMSE                    | 13.65 bu/acre                                      |
| Test R²                      | 0.845                                              |
| Test MAPE                    | 6.59%                                              |
| Prediciton Interval Coverage (95%)            | 95.0%                                              |
| Prediction Interval Width (mean)              | 57.31 bu/acre                                      |
| Model Inference Latency      | ~3-7ms (server-side, from API response)            |
| API Round-Trip Latency (p95) | 171.3ms (external, includes network)               |
| Training rows                | 20,576 (chronological split, years 2000–2014)      |

## Drift Monitoring

Data drift is monitored using Evidently AI, comparing incoming data against
a frozen reference distribution from training data.

**Event-driven (primary):** Automatically triggered after every real USDA
data refresh (`ingest.py --api-key ...`), since this is the actual point
where the input distribution could meaningfully shift — a new growing
season, revised acreage figures, etc.

**Scheduled (safety net):** A weekly cron job on the EC2 instance also runs
the check, primarily to demonstrate reliable automated scheduling and catch
any upstream pipeline issues between data refreshes — though corn yield
data itself only updates a few times per year, so this is intentionally a
lighter-weight backstop rather than the primary trigger.

```bash
python scripts/create_reference_dataset.py       # one-time setup
python src/monitoring/drift_report.py --current <new_data.csv>   # manual run
```

Detection validated with `scripts/inject_test_drift.py`.

## Known Limitations
- **Weather/soil features** (precipitation, temperature, soil pH) are only available in 
  synthetic demo data. Real USDA Quickstats data does not include these — would require 
  merging NOAA weather data and USDA NRCS soil survey data (planned for v2).
- **Harvested acreage** was excluded due to conflicting Survey vs Census values in USDA's 
  data with no reliable disambiguating field.

## Note on CI/CD
The GitHub Actions pipeline (`.github/workflows/ci.yml`) trains a quick model on 
synthetic demo data purely to verify the pipeline runs end-to-end 
(data → features → train → serve → Docker build). This keeps CI fast and 
independent of any API keys or credentials.

The production model referenced in the Results section below was trained
separately on real USDA Quickstats data (30,416 rows, county-level,
2000–2025) and is not what CI builds. Evaluated with a chronological train/val/test split, so
reported metrics reflect forward-looking generalization to unseen
future years. To reproduce the real model locally:

```bash
python src/data/ingest.py --api-key YOUR_USDA_KEY
python src/features/build_features.py
python src/models/train.py --run-name "production-model"
```
