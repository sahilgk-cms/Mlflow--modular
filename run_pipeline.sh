#!/bin/bash
echo Starting MLflow server...
docker compose up -d

echo "Building training image..."
docker build -t ap_training .

echo "Running training pipeline..."

docker run -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ap_training