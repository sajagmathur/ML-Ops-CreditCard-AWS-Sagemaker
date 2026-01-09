"""
SageMaker-compatible training script with MLflow logging
- Reads data from /opt/ml/input/data/train
- Saves model + metrics to /opt/ml/model
- Logs metrics & model to SageMaker Default MLflow
"""

# -----------------------------
# Imports
# -----------------------------
import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# SageMaker paths (DO NOT CHANGE)
# -----------------------------
INPUT_DIR = "/opt/ml/input/data/train"
OUTPUT_DIR = "/opt/ml/model"

# -----------------------------
# MLflow configuration
# -----------------------------
MLFLOW_EXPERIMENT_NAME = "creditcard-fraud-experiment"

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not mlflow_tracking_uri:
    raise RuntimeError("❌ MLFLOW_TRACKING_URI not set")

mlflow.set_tracking_uri(mlflow_tracking_uri)

# Create or set experiment
if not mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME):
    mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)

mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

print(f"✅ MLflow Tracking URI: {mlflow_tracking_uri}")
print(f"✅ MLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")

# -----------------------------
# Load data
# -----------------------------
data_path = os.path.join(INPUT_DIR, "Training.csv")
print(f"📥 Loading data from {data_path}")

data = pd.read_csv(data_path)

X = data.drop("Class", axis=1)
y = data["Class"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train + Log with MLflow
# -----------------------------
with mlflow.start_run(run_name="creditcard_training") as run:
    run_id = run.info.run_id
    print(f"🚀 MLflow Run ID: {run_id}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("📊 Metrics:", metrics)

    # Log metrics to MLflow
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))

    # Log model to MLflow (this creates model.pkl in MLflow artifacts)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    # -----------------------------
    # Save SageMaker artifacts
    # -----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(OUTPUT_DIR, "model.pkl"))

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print("✅ Model & metrics saved to /opt/ml/model")

print("🎉 Training + MLflow logging completed successfully")
