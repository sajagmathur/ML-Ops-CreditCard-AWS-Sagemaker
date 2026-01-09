import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep

# -----------------------------
# SageMaker session & role
# -----------------------------
session = sagemaker.Session()
role = "arn:aws:iam::075960506214:role/service-role/AmazonSageMaker-ExecutionRole-20251229T181530"

# -----------------------------
# Define SKLearn Estimator
# -----------------------------
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="s3://mlops-creditcard-sagemaker/prod_codes/source_dir.tar.gz",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    output_path="s3://mlops-creditcard-sagemaker/prod_outputs/trained_model/"
)

# -----------------------------
# Define TrainingStep
# -----------------------------
train_step = TrainingStep(
    name="TrainCreditCardModel",
    estimator=sklearn_estimator,
    inputs={
        "train": TrainingInput(
            s3_data="s3://mlops-creditcard-sagemaker/data/raw/",
            content_type="text/csv"
        )
    }
)

# -----------------------------
# Build Pipeline
# -----------------------------
pipeline = Pipeline(
    name="CreditCardTrainingPipeline",
    steps=[train_step],
    sagemaker_session=session
)

# -----------------------------
# Create or update pipeline
# -----------------------------
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print(f"✅ Training pipeline started: {execution.arn}")
