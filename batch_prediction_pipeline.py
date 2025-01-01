import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.v2.dsl import component

# TODO: Update the following values with your specific project information
PROJECT_ID = "bitstrapped-gpu-testing"  # Update with your project ID
BUCKET_NAME = "subtlemedical-demo-bucket"  # Update with your bucket name
LOCATION = "asia-southeast1"
BUCKET_URI = f"gs://{BUCKET_NAME}"

# https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-console
SERVICE_ACCOUNT = "id-72421690066-compute@bitstrapped-gpu-testing.iam.gserviceaccount.com"
API_ENDPOINT = "{}-aiplatform.googleapis.com".format(LOCATION)
PIPELINE_ROOT = "{}/pipeline_root/intro".format(BUCKET_URI)
MODEL_ID = "1553187717582422016"
MODEL_LOCATION = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}"
JSONL_FILE_URI = f"{BUCKET_URI}/subtle/batch_prediction/input_data.jsonl"
MACHINE_TYPE = "n1-standard-16"
# Default size 100GB for batch prediction
# https://stackoverflow.com/questions/69978953/vertex-ai-custom-prediction-vs-google-kubernetes-engine

aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, location=LOCATION)


@component(
    packages_to_install=['google-cloud-aiplatform'],
    base_image='python:3.9',
    output_component_file='deploy_model.yaml',
)
def batch_predict(
        model_uri: str,
        service_account: str,
        bucket_name: str,
        project: str, location: str, jsonl_file: list,
        machine_type: str
):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=location)

    uploaded_model = aiplatform.Model(model_uri)
    uploaded_model.batch_predict(
        instances_format="jsonl",
        predictions_format="jsonl",
        job_display_name="batch-prediction",
        gcs_source=jsonl_file,
        gcs_destination_prefix=f"{bucket_name}/batch_prediction/",
        starting_replica_count=1,
        max_replica_count=1,
        machine_type=machine_type,
        service_account=service_account,
    )


@dsl.pipeline(
    name="batch_prediction",
    description="Batch prediction",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline():
    batch_predict_task = batch_predict(
        model_uri=MODEL_LOCATION,
        service_account=SERVICE_ACCOUNT,
        bucket_name=BUCKET_URI, project=PROJECT_ID, location=LOCATION,
        jsonl_file=[JSONL_FILE_URI], machine_type=MACHINE_TYPE
    ).set_display_name("Batch Prediction")


from kfp.v2 import compiler  # noqa: F811

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="batch_prediction.json"
)

DISPLAY_NAME = "batch_prediction"

job = aip.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path="batch_prediction.json",
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False
)

job.run()
