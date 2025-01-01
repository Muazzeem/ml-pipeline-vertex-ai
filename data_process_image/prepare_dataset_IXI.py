import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as imresize

parser = argparse.ArgumentParser(description="Preprocess data and save to numpy arrays.")

parser.add_argument(
    '--data_source', default="bucket-name/folder-name",
    type=str, help="The bucket where the raw data persists"
)  # Define command-line argument for the source data bucket

parser.add_argument(
    '--data_destination', default="bucket-name/folder-name",
    type=str, help="The bucket where the processed data will be stored"
)  # Define command-line argument for the destination data bucket

args = parser.parse_args()


def download_folder(raw_data, folder):
    """
    Args:
        raw_data:   # The Google Storage bucket where the raw data is located.
        folder:     # The local folder where data will be downloaded.

    Returns:        # No explicit return value, as it downloads data to the specified folder.
    """
    # Construct the command to download data from Google Storage to the local folder.
    download_command = f"gsutil -m cp -r gs://{raw_data}/{folder}/* {folder}"

    # Execute the download command.
    os.system(download_command)


def upload(folder):
    """
    Args:
        folder: Construct the command to upload data from the local folder to a specified Google Storage bucket.

    Returns:    # No explicit return value, as it downloads data to the specified folder.

    """
    upload_command = f"gsutil -m cp -r {folder} gs://{args.data_destination}"

    # Execute the upload command.
    os.system(upload_command)


# Preprocessing: divide each case by the mean
def preprocess_data(data, norm=True, resize=256):
    if norm:
        data = data / data.mean()
    if data.shape[0] != resize:
        zfactor = resize / data.shape[0]
        data = imresize(data, (zfactor, zfactor, 1.0))
    return data


def process_case(case_dir, save_case_dir):
    t1 = preprocess_data(nib.load(glob(f"{case_dir}/*T1.nii.gz")[0]).get_fdata())
    t2 = preprocess_data(nib.load(glob(f"{case_dir}/*T2.nii.gz")[0]).get_fdata())
    pd = preprocess_data(nib.load(glob(f"{case_dir}/*PD.nii.gz")[0]).get_fdata())
    mra_file = glob(f"{case_dir}/*MRA.nii.gz")

    if len(mra_file) == 0:
        print(f'Skipping {case}. No MRA found')
        return

    mra = preprocess_data(nib.load(mra_file[0]).get_fdata())
    n_slices = t2.shape[-1]
    print(save_dir)
    for idx in range(skip_tb, n_slices - skip_tb):
        slice_i = np.stack([t1[:, :, idx], t2[:, :, idx], pd[:, :, idx], mra[:, :, idx]], axis=0)
        fn = os.path.join(save_case_dir, f"{idx:03d}.npy")
        np.save(fn, slice_i)


skip_tb = 0

folder = "train"


def init():
    os.makedirs(folder, exist_ok=True)  # Create the 'folder' if it doesn't exist, or do nothing if it already exists.


init()

save_dir = f'pre-processed/{folder}'
os.makedirs(save_dir, exist_ok=True)
download_folder(raw_data=args.data_source, folder=folder)
cases = sorted([c.split('/')[-1] for c in glob(f"{folder}/IXI*")])
# cases = sorted([c.split('/')[-1] for c in glob(f"{folder}/IXI*") if 'IOP' in c])
# cases = ['IXI002-Guys-0828']
for case in cases:
    save_case_dir = os.path.join(save_dir, case)
    os.makedirs(save_case_dir, exist_ok=True)
    process_case(f"{folder}/{case}", save_case_dir)
upload('pre-processed')
