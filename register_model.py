"""
Registers SageMaker-trained model into MLflow (Default MLflow App)

- Downloads model.tar.gz from S3
- Extracts model.pkl and metrics.json
- Logs metrics and model to MLflow
- Registers model as challenger
"""

# ============================================================
# 🔴 MUST COME FIRST — before importing mlflow / sklearn etc.
# ============================================================
import argparse
import os
import sys
import tempfile
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--MLFLOW_REQUIREMENTS_S3_URI",
    type=str,
    required=False,
    help="Optional S3 URI to requirements.txt (must include mlflow)",
)
parser.add_argument(
    "--MODEL_TAR_S3_URI",
    type=str,
    required=True,
    help="S3 URI of model.tar.gz from training step",
)

args, _ = parser.parse_known_args()

# -----------------------------
# Install dependencies before imports
# -----------------------------
if args.MLFLOW_REQUIREMENTS_S3_URI:
    print("📦 Installing dependencies before imports...")

    import boto3  # boto3 is available in SageMaker base images

    tmp_req_dir = tempfile.mkdtemp()
    local_req_path = os.path.join(tmp_req_dir, "requirements.txt")

    if args.MLFLOW_REQUIREMENTS_S3_URI.startswith("s3://"):
        bucket, key = args.MLFLOW_REQUIREMENTS_S3_URI.replace("s3://", "").split("/", 1)
        boto3.client("s3").download_file(bucket, key, local_req_path)
    else:
        local_req_path = args.MLFLOW_REQUIREMENTS_S3_URI

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", local_req_path])

# ============================================================
# ✅ SAFE TO IMPORT THIRD-PARTY LIBS NOW
# ============================================================
import json
import tarfile
from datetime import datetime

import boto3
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
S3_BUCKET = "mlops-creditcard-sagemaker"

MLFLOW_EXPERIMENT_NAME = "creditcard-fraud-experiment"
MLFLOW_MODEL_NAME = "creditcard-fraud-model"

MODEL_TAR_S3_URI = args.MODEL_TAR_S3_URI

print("📌 MODEL_TAR_S3_URI:", MODEL_TAR_S3_URI)
print("📌 MLFLOW_REQUIREMENTS_S3_URI:", args.MLFLOW_REQUIREMENTS_S3_URI)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_registry_uri(mlflow.get_tracking_uri())

if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
    mlflow.create_experiment(
        name=MLFLOW_EXPERIMENT_NAME,
        artifact_location=f"s3://{S3_BUCKET}/prod_outputs/mlflow",
    )

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -----------------------------
# Helpers
# -----------------------------
def parse_s3_uri(uri: str):
    uri = uri.replace("s3://", "")
    bucket, key = uri.split("/", 1)
    return bucket, key

# -----------------------------
# Download model.tar.gz
# -----------------------------
s3 = boto3.client("s3")
bucket, key = parse_s3_uri(MODEL_TAR_S3_URI)

tmp_dir = tempfile.mkdtemp()
local_tar_path = os.path.join(tmp_dir, "model.tar.gz")

print(f"📦 Downloading model artifact: s3://{bucket}/{key}")
s3.download_file(bucket, key, local_tar_path)

# -----------------------------
# Extract artifacts
# -----------------------------
with tarfile.open(local_tar_path, "r:gz") as tar:
    tar.extractall(tmp_dir)

model_path = os.path.join(tmp_dir, "model.pkl")
metrics_path = os.path.join(tmp_dir, "metrics.json")

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ model.pkl not found in model.tar.gz")

if not os.path.exists(metrics_path):
    raise FileNotFoundError("❌ metrics.json not found in model.tar.gz")

model = joblib.load(model_path)
with open(metrics_path) as f:
    metrics = json.load(f)

print("✅ Extracted model & metrics")

# -----------------------------
# Log to MLflow
# -----------------------------
run_name = f"register_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as run:
    run_id = run.info.run_id

    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))

    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.log_artifact(metrics_path, artifact_path="metrics")

print("✅ Logged model and metrics to MLflow")

# -----------------------------
# Register model as challenger
# -----------------------------
client = MlflowClient()
model_uri = f"runs:/{run_id}/model"

result = client.register_model(
    name=MLFLOW_MODEL_NAME,
    source=model_uri,
)

client.set_model_version_tag(MLFLOW_MODEL_NAME, result.version, "role", "challenger")
client.set_model_version_tag(MLFLOW_MODEL_NAME, result.version, "status", "staging")

print("🏷️ Model registered in MLflow")
print("Model Name:", MLFLOW_MODEL_NAME)
print("Version:", result.version)
print("Run ID:", run_id)
print("===== END MLflow registration =====")