import os
import random

import gcsfs
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import IXISingleDataset, RandomGeneratorIXI

# Load environment variables if the .env file exists
if os.path.exists(".env"):
    load_dotenv(".env")

# Check if CUDA is available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set project and file paths using environment variables
PROJECT_ID = os.getenv("PROJECT_ID", "project-id")
STORE_FILES_PATH = os.getenv("STORE_FILES_PATH", "subtlemedical-demo-bucket/predict_result")

# Create a GCS filesystem connection
fs = gcsfs.GCSFileSystem(project=PROJECT_ID)


# Function to split data
def split_data(data, inputs, targets):
    contrast_input = inputs
    contrast_output = targets
    img_inputs = [data[i].detach().to(device) for i in contrast_input]
    img_targets = [data[i].detach().to(device) for i in contrast_output]
    return img_inputs, img_targets, contrast_input, contrast_output


# Function to save data to GCS
def save_data(folder_name, output_arrays):
    save_dir = 'prediction/result'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{folder_name}.npy"

    fn = os.path.join(save_dir, file_name)
    np.save(fn, output_arrays)

    # Upload the saved file to GCS
    fs.put(lpath=f"{save_dir}/{file_name}", rpath=f"{STORE_FILES_PATH}/{file_name}")
    print(f"Uploaded {file_name}")

    return f"gcs://{STORE_FILES_PATH}/{file_name}"


# Function to evaluate the model on IXI dataset
def eval_ixi(model, inputs, targets, dataloader):
    print('eval_ixi')

    model.eval()

    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            img_data = [d.detach().to(device) for d in data]
            img_inputs, img_targets, contrast_input, contrast_output = split_data(img_data, inputs, targets)
            print('img_inputs & contrast_input & output',
                  len(img_inputs), len(contrast_input), len(contrast_output))
            img_outputs, _, _ = model(img_inputs, contrast_input, contrast_output)
            output_arrays = []
            for i, outputs in enumerate(img_outputs):
                output_imgs = outputs.detach().cpu().numpy()
                output_arrays.append(output_imgs)
    return output_arrays, None, None


# Load data & iterate over elements for inference
def evaluator_ixi(args, model, input_combination, targets, folder_name):
    batch_size = args.batch_size * args.n_gpu
    db = IXISingleDataset(
        base_dir=f"{args.root_path}/pre-process/{folder_name}",
        transform=transforms.Compose(
            [RandomGeneratorIXI(flip=False, scale=None, n_contrast=4 if args.mra_synth else 3)])
    )

    print("The length of the data set is: {}".format(len(db)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    batch_size = 6  # Temporary value

    # Create a DataLoader for the dataset
    dataloader = DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    if len(db) > 1:
        output_arrays, _, _ = eval_ixi(model, input_combination, targets, dataloader)
        result = save_data(folder_name=folder_name, output_arrays=output_arrays)
        return result
    else:
        return "skip"
