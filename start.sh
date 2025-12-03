#!/bin/bash

mlflow server \
  --backend-store-uri "mysql+pymysql://admin:Intel%4021@52.70.125.119:3306/mlflow_db_latest" \
  --artifacts-destination "s3://intel-mlflow" \
  --default-artifact-root "s3://intel-mlflow" \
  --host 0.0.0.0 \
  --port 8000
