## Subtlemedical train pipeline

## Installing and Configuring Google Cloud SDK (gcloud)

This guide will walk you through the steps to install and configure the Google Cloud SDK (gcloud) on your local machine.

### Installation

Follow this [documentation](https://cloud.google.com/sdk/docs/install) to install the Google Cloud SDK.

## Initialize the SDK

After the installation is complete, you need to initialize
the SDK by running the following command in your terminal:

```shell
gcloud init
```

Configuration
During the `gcloud init` process, you will be prompted to log in with your Google Account and select a project. Follow
the prompts to complete the configuration. If you need to configure additional settings or switch projects later, you
can use the following commands:

```shell
gcloud config set project PROJECT_ID: Set the default project.
```

For example: `gcloud config set project bitstrapped-gpu-testing`

```shell
gcloud config set compute/zone ZONE: Set the default compute zone.
```

```shell
gcloud config set compute/region REGION: Set the default compute region.
```

```shell
gcloud auth application-default login
```

## Create a service account and set permission for run pipeline

1. Navigate to the `permission.sh` script's directory.
2. Change its permissions using the chmod command.
```shell
chmod +x permission.sh
```
3. Run the Script by using `./permission.sh`

## Build and push the artifact registry

1. Open a terminal and navigate to the directory where you have the `cloudbuild.yaml` file.

2. Edit the `cloudbuild.yaml` file and update the following substitutions:

`_PROJECT`: Replace this with your Google Cloud project name.

`_LOCATION`: Replace this with your desired Google Cloud region/zone.

Then run this command:

```shell
gcloud builds submit . 
```

Or you can pass your project name and location using this command

```shell
gcloud builds submit --substitutions=_PROJECT="<project_name>",_LOCATION="<location>" .
```

**Note:** Ensure that your computer has the required Google credentials for authentication.

## Run training pipeline

### Prerequisites

Before running the training pipeline, ensure that you have the necessary dependencies installed.

### Installation

```bash
pip install -r requirements.txt
```

### Running the Training Pipeline

To execute the training pipeline, follow these steps:

1. Open your terminal.
2. Navigate to the project directory.
3. Run the training_pipeline.py script using the following command:

```bash
python3 training_pipeline.py
```

**Note:** Ensure that your computer has the required Google credentials for authentication.

## Run batch prediction pipeline

### jsonl file template for batch prediction

1. Create a new JSONL file (JSON Lines format) using a text editor or code editor of your choice.
2. Add the following content to the JSONL file, replacing `<bucket_name>` with your actual GCS bucket name:

```json
{"gcs_path": "gs://<bucket_name>/raw/train/IXI002-Guys-0828", "input_combination": [1, 2], "targets": [0]}
```

3. Save the file with a meaningful name, such as `input_data.jsonl`.

### Prerequisites

Before running the batch prediction pipeline, ensure that you have the necessary dependencies installed.

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

To execute the training pipeline, follow these steps:

1. Open your terminal.
2. Navigate to the project directory.
3. Run the `batch_prediction_pipeline.py` script using the following command:

```bash
python3 batch_prediction_pipeline.py
```

**Note:** Ensure that your computer has the required Google credentials for authentication.

### Reference

* [Create Artifact Registry](https://cloud.google.com/artifact-registry/docs/repositories/create-repos)
* [Build and Push Artifact Registry](https://cloud.google.com/sdk/gcloud/reference/builds/submit)
