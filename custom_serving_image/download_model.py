import os

import gcsfs
from dotenv import load_dotenv

# Check if a .env file exists and load its contents if it does
if os.path.exists(".env"):
    load_dotenv(".env")

# Get the PROJECT_ID environment variable or use a default value if not set
PROJECT_ID = os.getenv("PROJECT_ID", "project-id")

# Create a Google Cloud Storage (GCS) file system using the specified project ID
fs = gcsfs.GCSFileSystem(project=PROJECT_ID)


# Define a function to download the latest model from GCS
def download_latest_model(model_dir, local_dir):
    # Print a message indicating the download process
    print(f"Downloading model from: {model_dir} to {local_dir}")

    # Use the GCSFileSystem to download the model from GCS to the local directory
    fs.get(rpath=model_dir, lpath=local_dir)
    print("Model downloaded")
