"""
Registers SageMaker-trained model into MLflow (Default MLflow App)
- Downloads model.tar.gz from S3
- Extracts model + metrics
- Logs artifacts to MLflow
- Registers model as challenger
"""

import os
import json
import tarfile
import tempfile
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

# Passed via Step Functions / environment
MODEL_TAR_S3_URI = os.environ.get(
    "MODEL_TAR_S3_URI",
    f"s3://{S3_BUCKET}/prod_outputs/trained_model/"
)

MLFLOW_EXPERIMENT_NAME = "creditcard-fraud-experiment"
MLFLOW_MODEL_NAME = "creditcard-fraud-model"

# -----------------------------
# MLflow setup (Default MLflow App)
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
# Download model.tar.gz from S3
# -----------------------------
s3 = boto3.client("s3")

def parse_s3_uri(uri):
    uri = uri.replace("s3://", "")
    bucket, key = uri.split("/", 1)
    return bucket, key

bucket, prefix = parse_s3_uri(MODEL_TAR_S3_URI)

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
candidates = [o for o in response.get("Contents", []) if o["Key"].endswith("model.tar.gz")]

if not candidates:
    raise FileNotFoundError("No model.tar.gz found in S3 path")

model_obj = sorted(candidates, key=lambda x: x["LastModified"], reverse=True)[0]
model_key = model_obj["Key"]
print(f"📦 Using model artifact: s3://{bucket}/{model_key}")

tmp_dir = tempfile.mkdtemp()
local_tar = os.path.join(tmp_dir, "model.tar.gz")
s3.download_file(bucket, model_key, local_tar)

# -----------------------------
# Extract model and metrics
# -----------------------------
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(tmp_dir)

model_path = os.path.join(tmp_dir, "model.pkl")
metrics_path = os.path.join(tmp_dir, "metrics.json")

if not os.path.exists(model_path) or not os.path.exists(metrics_path):
    raise FileNotFoundError("model.pkl or metrics.json not found in model.tar.gz")

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

    # log metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))

    # log model
    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.log_artifact(metrics_path, artifact_path="metrics")

print("✅ Logged artifacts to MLflow")

# -----------------------------
# Register as challenger
# -----------------------------
model_uri = f"runs:/{run_id}/model"
result = mlflow.register_model(model_uri=model_uri, name=MLFLOW_MODEL_NAME)
client = MlflowClient()

client.set_model_version_tag(name=MLFLOW_MODEL_NAME, version=result.version, key="role", value="challenger")
client.set_model_version_tag(name=MLFLOW_MODEL_NAME, version=result.version, key="status", value="staging")

print("🏷️ Model registered & tagged")
print("Model:", MLFLOW_MODEL_NAME)
print("Version:", result.version)
print("Run ID:", run_id)
print("===== END MLflow logging =====")
