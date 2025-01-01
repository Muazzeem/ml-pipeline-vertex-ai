import os
from argparse import Namespace

import torch
from dotenv import load_dotenv
from flask import Flask, Response, request, jsonify

from download_model import download_latest_model
from model_inference import predict_with_model, load_model
from pre_process_data import download_raw_and_process

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    load_dotenv(".env")

# Default values for environment variables
BUCKET_NAME = os.getenv("BUCKET_NAME", "subtlemedical-demo-bucket")
MODEL_NAME = os.getenv("MODEL_NAME", "epoch_1.pth")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "model-latest/")
GCS_MODEL_DIR = os.getenv("GCS_MODEL_DIR", "subtlemedical-demo-bucket/models/latest/epoch_1.pth")
# Create a Flask app
app = Flask(__name__)


def init_model():
    # Download the latest model if it doesn't exist
    if not os.path.isdir('model-latest'):
        download_latest_model(
            model_dir=GCS_MODEL_DIR,
            local_dir=LOCAL_MODEL_PATH,
        )
    else:
        print("Model already downloaded")


init_model()

# Define command-line arguments using Namespace for better organization
args = Namespace(
    root_path='.',
    cfg='configs/mmt_ixi.yml',
    dataset='IXI',
    batch_size=32,
    n_gpu=0,
    k=2,
    zero_gad=False,
    ckpt=MODEL_NAME,
    model_path=LOCAL_MODEL_PATH,
    vis=True,
    seg=False,
    masked=False,
    vis_dir='.',
    seed=1234,
    n_contrast=3,
    save_enc_out=False,
    mra_synth=False,
    no_cross_contrast_attn=False
)

data_dir = args.root_path
save_dir = f"{data_dir}/pre-process/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model once outside the loop
loaded_model = load_model(args, device)


@app.route("/isalive")
def is_alive():
    status_code = Response(status=200)
    return status_code


# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    req_json = request.get_json()

    predictions = []

    for instance in req_json['instances']:
        stripped_string = instance["gcs_path"].replace("gs://", "")
        path_segments = stripped_string.split('/')
        folder_name = path_segments[-1]
        pre_process_dir = download_raw_and_process(gcs_file_path=stripped_string, data_dir=data_dir, save_dir=save_dir)

        master_list = predict_with_model(
            loaded_model, args, instance['input_combination'], instance["targets"], folder_name=folder_name
        )
        predictions.append({folder_name: master_list})

        # Clean up - delete directory after inference
        import shutil
        npy_array_dir = f"{data_dir}/{folder_name}"
        if os.path.isdir(npy_array_dir):
            shutil.rmtree(npy_array_dir)

    return jsonify({
        "predictions": predictions
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
