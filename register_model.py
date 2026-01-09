"""
Registers SageMaker-trained model into MLflow (Default MLflow App)

- Downloads model.tar.gz from S3
- Extracts model.pkl and metrics.json
- Logs metrics and model to MLflow
- Registers model as challenger
"""

import os
import tempfile
import subprocess

# -----------------------------
# Install dependencies from S3
# -----------------------------
requirements_s3_uri = os.environ.get("MLFLOW_REQUIREMENTS_S3_URI")
if requirements_s3_uri:
    tmp_dir = tempfile.mkdtemp()
    local_req_path = os.path.join(tmp_dir, "requirements.txt")

    if requirements_s3_uri.startswith("s3://"):
        import boto3
        bucket, key = requirements_s3_uri.replace("s3://", "").split("/", 1)
        print(f"📦 Downloading requirements.txt from s3://{bucket}/{key}")
        boto3.client("s3").download_file(bucket, key, local_req_path)
    else:
        local_req_path = requirements_s3_uri

    print("📦 Installing Python dependencies...")
    subprocess.check_call(["pip", "install", "--upgrade", "pip"])
    subprocess.check_call(["pip", "install", "-r", local_req_path])

# -----------------------------
# Now import MLflow and others
# -----------------------------
import json
import tarfile
import joblib
import boto3
import mlflow
import mlflow.sklearn
from datetime import datetime
from mlflow.tracking import MlflowClient

# -----------------------------
# Configuration
# -----------------------------
S3_BUCKET = "mlops-creditcard-sagemaker"

MODEL_TAR_S3_URI = os.environ.get("MODEL_TAR_S3_URI")
if not MODEL_TAR_S3_URI:
    raise ValueError("MODEL_TAR_S3_URI environment variable not set")

MLFLOW_EXPERIMENT_NAME = "creditcard-fraud-experiment"
MLFLOW_MODEL_NAME = "creditcard-fraud-model"

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_registry_uri(mlflow.get_tracking_uri())

if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
    mlflow.create_experiment(
        name=MLFLOW_EXPERIMENT_NAME,
        artifact_location=f"s3://{S3_BUCKET}/prod_outputs/mlflow"
    )

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -----------------------------
# Helpers
# -----------------------------
def parse_s3_uri(uri):
    uri = uri.replace("s3://", "")
    bucket, key = uri.split("/", 1)
    return bucket, key

# -----------------------------
# Download model.tar.gz
# -----------------------------
s3 = boto3.client("s3")

bucket, key = parse_s3_uri(MODEL_TAR_S3_URI)

print(f"📦 Downloading model artifact: s3://{bucket}/{key}")

tmp_dir = tempfile.mkdtemp()
local_tar = os.path.join(tmp_dir, "model.tar.gz")
s3.download_file(bucket, key, local_tar)

# -----------------------------
# Extract artifacts
# -----------------------------
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(tmp_dir)

model_path = os.path.join(tmp_dir, "model.pkl")
metrics_path = os.path.join(tmp_dir, "metrics.json")

if not os.path.exists(model_path):
    raise FileNotFoundError("model.pkl not found in model.tar.gz")

if not os.path.exists(metrics_path):
    raise FileNotFoundError("metrics.json not found in model.tar.gz")

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

    # Log metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Log metrics file
    mlflow.log_artifact(metrics_path, artifact_path="metrics")

print("✅ Logged model and metrics to MLflow")

# -----------------------------
# Register model as challenger
# -----------------------------
client = MlflowClient()
model_uri = f"runs:/{run_id}/model"

result = client.register_model(
    name=MLFLOW_MODEL_NAME,
    source=model_uri
)

client.set_model_version_tag(
    name=MLFLOW_MODEL_NAME,
    version=result.version,
    key="role",
    value="challenger"
)

client.set_model_version_tag(
    name=MLFLOW_MODEL_NAME,
    version=result.version,
    key="status",
    value="staging"
)

print("🏷️ Model registered in MLflow")
print("Model Name:", MLFLOW_MODEL_NAME)
print("Version:", result.version)
print("Run ID:", run_id)
print("===== END MLflow registration =====")
