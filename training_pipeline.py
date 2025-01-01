# Import necessary modules
from datetime import datetime
from typing import NamedTuple

import google.cloud.aiplatform as aip
import kfp
from kfp import compiler, dsl
from kfp.dsl import ContainerSpec
from kfp.dsl import container_component

# TODO: Update the following values with your specific project information
PROJECT_ID = "bitstrapped-gpu-testing"  # Update with your project ID
BUCKET_NAME = "subtlemedical-demo-bucket"  # Update with your bucket name
REPO_NAME = "subtlemedical-ml-train"  # Update with your repository name
LOCATION = "asia-southeast1"
BUCKET_URI = f"gs://{BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/subtle/models"
VERSION = datetime.now().strftime("%Y%m%d%H%M%S")
MODEL_NAME = "model"
MODEL_DISPLAY_NAME = f"{MODEL_NAME}-{VERSION}"

PRE_PROCESS_MACHINE_TYPE = "e2-highmem-16"

SERVING_MIN_REPLICA_COUNT = 1
SERVING_MAX_REPLICA_COUNT = 1
SERVING_HEALTH_ROUTE = "/isalive"
SERVING_PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORT = [{"containerPort": 8080}]
DEPLOY_COMPUTE = "e2-highmem-16"

"""
Disk size is important as we save all data in a single drive.
Please kindly increase or decrease based on your data volume. AFTER PREPROCESSING, 37GB of 
raw data is converted to 130 GB of data. So, we have 
300 GB of size

"""
BOOT_DISK_SIZE_IN_GB = 300
# Initialize the AI Platform project
aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, location=LOCATION)


# Function to get a timestamp

def get_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M%S")


# Generate a timestamp
TIMESTAMP = get_timestamp()

# Define some constants
MACHINE_TYPE = "a2-highgpu-4g"
NUMBER_OF_GPU = 4
BATCH_SIZE = 24
MAX_EPOCHS = 50
MAX_EPOCHS = 1  # For testing
NUMBER_OF_INPUT = 2
NUMBER_OF_CONTRAST = 3

# Define trainer arguments
TRAINER_ARGS = [
    "--root_path", "/subtlemedical-ml-pipeline",
    "--data_path", f"{BUCKET_NAME}/pre-processed",  # Pre-processed data path
    "--dataset", "IXI", "--project", PROJECT_ID,
    "--n_gpu", str(NUMBER_OF_GPU), "--batch_size", str(BATCH_SIZE),
    "--max_epochs", str(MAX_EPOCHS)
]

# Define machine specifications
MACHINE_SPEC_FOR_TRAINING = {
    "machineType": MACHINE_TYPE,
    "accelerator_type": aip.gapic.AcceleratorType.NVIDIA_TESLA_A100,
    "accelerator_count": NUMBER_OF_GPU,
}

# Define Docker image URIs
PRE_PROCESS_IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/subtlemedical-data-preprocess/preprocess:latest"
TRAIN_IMAGE = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/train-model:latest"
SERVE_IMAGE_URI = f"{LOCATION}-docker.pkg.dev/{PROJECT_ID}/subtlemedical-prediction/vertexai:latest"

# Define the working directory
MODEL_DIR = f"{PIPELINE_ROOT}/{TIMESTAMP}"


# Define a container component for preprocessing training data
@container_component
def pre_process_train_data():
    return ContainerSpec(
        image=PRE_PROCESS_IMAGE_URI,
        command=[
            'python3',
            'prepare_dataset_IXI.py',
        ],
        args=[
            "--data_source", f"{BUCKET_NAME}/raw",
            "--data_destination", BUCKET_NAME
        ],
    )


@dsl.component(
    packages_to_install=['google-cloud-aiplatform'],
    base_image='python:3.9',
    output_component_file='deploy_model.yaml',
)
def deploy_model(
        saved_model_path: str,
        project_id: str,
        location: str,
        model_name: str,
        serving_image_uri: str,
        predict_route: str, health_route: str,
):
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=location)

    # List all models in the Model Registry from this market
    DISPLAY_NAME = f'subtle-medical-{model_name}'
    models = aiplatform.Model.list(filter=('display_name={}').format(DISPLAY_NAME))
    # If one exists, log this model as a new version. If not, log this model with no parent
    if len(models) == 0:
        model_upload = aiplatform.Model.upload(
            display_name=DISPLAY_NAME,
            artifact_uri=saved_model_path,
            serving_container_image_uri=serving_image_uri,
            serving_container_predict_route=predict_route,
            serving_container_health_route=health_route
        )
    else:
        parent_model = models[0].resource_name
        model_upload = aiplatform.Model.upload(
            display_name=DISPLAY_NAME,
            parent_model=parent_model,
            artifact_uri=saved_model_path,
            serving_container_image_uri=serving_image_uri,
            serving_container_predict_route=predict_route,
            serving_container_health_route=health_route
        )

@dsl.component(
    packages_to_install=['google-cloud-aiplatform'],
    base_image='python:3.9',
    output_component_file='dataset_metadata.yaml'
)
def write_dataset_artifact(
        project: str,
        location: str,
        pre_process_data_destination: str
) -> NamedTuple(
    "Outputs",
    [
        ("dataset_artifact", str),  # Return parameters
        ("pre_processed_uri", str)
    ],
):
    import datetime
    from google.cloud.aiplatform.metadata.schema.system import artifact_schema
    pre_processed_uri = pre_process_data_destination
    system_artifact_schema = artifact_schema.Dataset(
        uri=pre_processed_uri,
        display_name='Dataset Artifact',
        metadata={
            'timestamp': str(datetime.datetime.now()),
            "pre_processed_uri": pre_processed_uri,
            "type": "IXI"
        }
    )

    # Create the dataset_artifact directly
    dataset_artifact = system_artifact_schema.create(project=project, location=location)
    return (dataset_artifact.name, pre_processed_uri)


@dsl.component(
    packages_to_install=['google-cloud-aiplatform', 'gcsfs'],
    base_image='python:3.9',
    output_component_file='model_metadata.yaml',
)
def create_model_metadata(
        model_metadata_uri: str,
        model_dir: str,
        artifact_id: str,
        project: str,
        location: str, bucket_name: str,
        max_epoch: int, k: int, n_contrast: int
):
    import datetime
    import gcsfs
    import json
    from google.cloud import aiplatform
    from google.cloud.aiplatform.metadata.schema.system import artifact_schema, execution_schema
    fs = gcsfs.GCSFileSystem(project=project)
    model_artifact = artifact_schema.Model(
        uri=model_metadata_uri,
        display_name='Model Artifact',
        metadata={
            'timestamp': str(datetime.datetime.now()),
            "model_uri": model_dir,
            "k": k, "n_contrast": n_contrast, "max_epoch": max_epoch
        }
    )
    model_artifact.create(project=project, location=location)

    model_info = fs.read_text(f'{bucket_name}/models/latest/logs/model_info.json')
    data = json.loads(model_info)
    metrics_artifact = artifact_schema.Metrics(
        uri=model_metadata_uri,
        display_name='Metrics Artifact',
        metadata={
            "timestamp": str(datetime.datetime.now()),
            "best_mse": data["mse"], "best_mae": data["mae"]
        }
    )
    metrics_artifact.create(project=project, location=location)
    dataset_artifact = aiplatform.Artifact.get(
        resource_id=artifact_id, project=project, location=location
    )
    with execution_schema.ContainerExecution(
            display_name='Execution',
            metadata=None,
            schema_version=None,
            description=None,
    ).create() as execution:
        execution.assign_input_artifacts([dataset_artifact])
        execution.assign_output_artifacts([model_artifact])

    with execution_schema.ContainerExecution(
            display_name='Metrics Execution',
            metadata={
                "timestamp": str(datetime.datetime.now())
            },
            schema_version=None,
            description=None,
    ).create() as metric_execution:
        metric_execution.assign_input_artifacts([model_artifact])
        metric_execution.assign_output_artifacts([metrics_artifact])


# Define the main pipeline
@kfp.dsl.pipeline(name="training-job")
def pipeline(
        project: str = PROJECT_ID,
):
    from google_cloud_pipeline_components.v1.custom_job import \
        CustomTrainingJobOp
    from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component

    # Create a custom training job from the preprocessing container
    custom_training_job = create_custom_training_job_from_component(
        pre_process_train_data,
        machine_type=PRE_PROCESS_MACHINE_TYPE,
        boot_disk_size_gb=BOOT_DISK_SIZE_IN_GB
    )
    data_processor = custom_training_job(
        project=PROJECT_ID,
        location=LOCATION,
    ).set_display_name("Process Train Data")

    create_dataset = write_dataset_artifact(
        pre_process_data_destination=f"{BUCKET_URI}/pre-processed/train/",
        project=PROJECT_ID, location=LOCATION
    ).after(data_processor).set_display_name("Dataset Artifact")

    # Define a custom training job
    custom_train_job = CustomTrainingJobOp(
        project=project,
        location=LOCATION,
        display_name="model-training",
        worker_pool_specs=[
            {
                "containerSpec": {
                    "args": TRAINER_ARGS,
                    "env": [{"name": "AIP_MODEL_DIR", "value": f"{MODEL_DIR}/"}],
                    "imageUri": TRAIN_IMAGE,
                },
                "replicaCount": "1",
                # https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus
                "machineSpec": MACHINE_SPEC_FOR_TRAINING,
                "diskSpec": {
                    "bootDiskSizeGb": BOOT_DISK_SIZE_IN_GB
                },
            }
        ]
    ).set_display_name("Train Model").after(create_dataset)

    model_metadata = create_model_metadata(
        model_metadata_uri=f"{MODEL_DIR}/",
        model_dir=f"{MODEL_DIR}/", artifact_id=create_dataset.outputs["dataset_artifact"],
        project=PROJECT_ID, location=LOCATION,
        k=NUMBER_OF_INPUT, n_contrast=NUMBER_OF_CONTRAST, max_epoch=MAX_EPOCHS,
        bucket_name=BUCKET_NAME
    ).set_display_name("Model Metadata").after(custom_train_job)

    # Deploy the model and run batch prediction
    model_upload_op = deploy_model(
        saved_model_path=f"{MODEL_DIR}/",
        project_id=PROJECT_ID,
        location=LOCATION,
        model_name=MODEL_NAME,
        serving_image_uri=SERVE_IMAGE_URI,
        predict_route=SERVING_PREDICT_ROUTE,
        health_route=SERVING_HEALTH_ROUTE,
    ).after(model_metadata).set_display_name("Deploy Model")


# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="training.json",
)

# Define the display name for the job
DISPLAY_NAME = "training-job"

# Create a pipeline job and run it
job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="training.json",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False
)

SERVICE_ACCOUNT = "pipeline-service-account@bitstrapped-gpu-testing.iam.gserviceaccount.com"

job.run(
    service_account=SERVICE_ACCOUNT
)
