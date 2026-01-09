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
    base_job_name="creditcard-training"
)

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
# 2️⃣ Register Model (MLflow)
# -----------------------------
register_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["bash"],
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
)

register_step = ProcessingStep(
    name="RegisterModelWithMLflow",
    processor=register_processor,
    code=None,  # override entrypoint
    job_arguments=[
        "-c",
        f"""
        set -e
        # download trained model artifact
        aws s3 cp {train_step.properties.ModelArtifacts.S3ModelArtifacts} /tmp/model.tar.gz

        # extract model & metrics
        mkdir -p /opt/ml/processing/model
        tar -xzf /tmp/model.tar.gz -C /opt/ml/processing/model

        # move to code folder
        cp -r /opt/ml/processing/model/* /opt/ml/processing/code/

        cd /opt/ml/processing/code

        # install requirements (including mlflow)
        pip install -r requirements.txt

        # register model with MLflow
        python register_model.py --MODEL_TAR_S3_URI /opt/ml/processing/code
        """
    ],
)

# -----------------------------
# 3️⃣ Build Pipeline
# -----------------------------
pipeline = Pipeline(
    name="CreditCardTrainingPipeline",
    steps=[train_step, register_step],
    sagemaker_session=session
)

# -----------------------------
# 4️⃣ Create / Update Pipeline
# -----------------------------
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print(f"✅ Pipeline started: {execution.arn}")
