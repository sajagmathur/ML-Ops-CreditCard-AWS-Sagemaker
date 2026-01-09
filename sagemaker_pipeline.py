from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
import sagemaker

# -----------------------------
# SageMaker session and role
# -----------------------------
sagemaker_session = sagemaker.Session()
sagemaker_role = "<YOUR_SAGEMAKER_EXECUTION_ROLE>"

# -----------------------------
# 1️⃣ Training Step
# -----------------------------
sklearn_estimator = SKLearn(
    entry_point="train.py",
    role=sagemaker_role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://mlops-creditcard-sagemaker/prod_outputs/trained_model/",
    base_job_name="creditcard-training",
)

training_step = TrainingStep(
    name="TrainCreditCardModel",
    estimator=sklearn_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://mlops-creditcard-sagemaker/data/raw/",
            content_type="text/csv"
        )
    }
)

# -----------------------------
# 2️⃣ Register Model Step
# -----------------------------
register_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["python3"],
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.large",
)

register_step = ProcessingStep(
    name="RegisterModelWithMLflow",
    processor=register_processor,
    code="register_model.py",
    job_arguments=[
        f"--MODEL_TAR_S3_URI={training_step.properties.ModelArtifacts.S3ModelArtifacts}",
        f"--MLFLOW_TRACKING_URI=default"  # Use SageMaker Default MLflow App
    ],
    inputs=[],
    outputs=[]
)

# -----------------------------
# 3️⃣ Champion Selection Step
# -----------------------------
champion_processor = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
    command=["python3"],
    role=sagemaker_role,
    instance_count=1,
    instance_type="ml.m5.large",
)

champion_step = ProcessingStep(
    name="ChampionSelection",
    processor=champion_processor,
    code="championselection.py",
    job_arguments=[
        f"--MLFLOW_TRACKING_URI=default",  # Default MLflow App on SageMaker
        f"--S3_BUCKET=mlops-creditcard-sagemaker"
    ],
    inputs=[],
    outputs=[]
)

# -----------------------------
# 4️⃣ Define the Pipeline
# -----------------------------
pipeline = Pipeline(
    name="CreditCardTrainingPipeline",
    steps=[training_step, register_step, champion_step],
    sagemaker_session=sagemaker_session
)
