#!/bin/bash
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate corn-yield
cd /home/ubuntu/corn_yield_prediction_pipeline
python src/monitoring/drift_report.py --current data/processed/corn_yield_features.csv \
  >> logs/drift_check.log 2>&1
