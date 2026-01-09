# -----------------------------
# MLflow logging
# -----------------------------
import mlflow
import mlflow.sklearn

MLFLOW_EXPERIMENT_NAME = "creditcard-fraud-experiment"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

with mlflow.start_run(run_name="training_run") as run:
    # Log model metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))
    
    # Log model artifact
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    # Save local metrics
    with open(os.path.join("/opt/ml/model", "metrics.json"), "w") as f:
        json.dump(metrics, f)

print("✅ Training metrics logged to MLflow")
