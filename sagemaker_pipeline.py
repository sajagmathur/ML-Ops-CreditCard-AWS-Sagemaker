import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.sklearn.processing import ScriptProcessor

# -----------------------------
# SageMaker session & role
# -----------------------------
session = sagemaker.Session()
role = "arn:aws:iam::075960506214:role/service-role/AmazonSageMaker-ExecutionRole-20251229T181530"

# -----------------------------
# 1️⃣ Training Estimator
# -----------------------------
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="s3://mlops-creditcard-sagemaker/prod_codes/source_dir.tar.gz",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    output_path="s3://mlops-creditcard-sagemaker/prod_outputs/trained_model/",
    base_job_name="creditcard-training",
)

train_step = TrainingStep(
    name="TrainCreditCardModel",
    estimator=sklearn_estimator,
    inputs={
        "train": TrainingInput(
            s3_data="s3://mlops-creditcard-sagemaker/data/raw/",
            content_type="text/csv",
        )
    },
)

# -----------------------------
# 2️⃣ Register Model (MLflow)
# -----------------------------
register_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    # Pass environment variable so register_model.py can install MLflow
    env={
        "MLFLOW_REQUIREMENTS_S3_URI": "s3://mlops-creditcard-sagemaker/prod_codes/requirements.txt"
    }
)

register_step = ProcessingStep(
    name="RegisterModelWithMLflow",
    processor=register_processor,
    code="s3://mlops-creditcard-sagemaker/prod_codes/register_model.py",
    job_arguments=[
        "--MODEL_TAR_S3_URI", train_step.properties.ModelArtifacts.S3ModelArtifacts
    ]
)

# -----------------------------
# 3️⃣ Champion Selection
# -----------------------------
champion_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["python3"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
)

champion_step = ProcessingStep(
    name="ChampionSelection",
    processor=champion_processor,
    code="s3://mlops-creditcard-sagemaker/prod_codes/championselection.py",
    depends_on=[register_step]  # ensures champion selection runs after registration
)

# -----------------------------
# 4️⃣ Build Pipeline
# -----------------------------
pipeline = Pipeline(
    name="CreditCardTrainingPipeline",
    steps=[train_step, register_step, champion_step],
    sagemaker_session=session,
)

# -----------------------------
# 5️⃣ Create / Update Pipeline
# -----------------------------
pipeline.upsert(role_arn=role)
execution = pipeline.start()

print(f"✅ Pipeline started: {execution.arn}")
