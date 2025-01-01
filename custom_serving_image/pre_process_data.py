import os
from glob import glob

import gcsfs
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from scipy.ndimage import zoom as imresize

if os.path.exists(".env"):
    load_dotenv(".env")

PROJECT_ID = os.getenv("PROJECT_ID", "default-project-id")

fs = gcsfs.GCSFileSystem(project=PROJECT_ID)


# Preprocessing: divide each case by the mean
def preprocess_data(data, norm=True, resize=256):
    if norm:
        data = data / data.mean()
    if data.shape[0] != resize:
        zfactor = resize / data.shape[0]
        data = imresize(data, (zfactor, zfactor, 1.0))
    return data


def process_case(case, data_dir, save_dir):
    t1 = preprocess_data(nib.load(glob(f"{data_dir}/{case}/*T1.nii.gz")[0]).get_fdata())
    t2 = preprocess_data(nib.load(glob(f"{data_dir}/{case}/*T2.nii.gz")[0]).get_fdata())
    pd = preprocess_data(nib.load(glob(f"{data_dir}/{case}/*PD.nii.gz")[0]).get_fdata())
    mra_file = glob(f"{data_dir}/{case}/*MRA.nii.gz")

    if len(mra_file) == 0:
        print(f'Skipping {case}. No MRA found')
        return

    mra = preprocess_data(nib.load(mra_file[0]).get_fdata())
    n_slices = t2.shape[-1]
    for idx in range(skip_tb, n_slices - skip_tb):
        slice_i = np.stack([t1[:, :, idx], t2[:, :, idx], pd[:, :, idx], mra[:, :, idx]], axis=0)
        fn = os.path.join(f"{save_dir}/{case}", f"{idx:03d}.npy")
        np.save(fn, slice_i)
    print(f"Completed {case}")


skip_tb = 0


def download_raw_and_process(gcs_file_path, data_dir, save_dir):
    print("Raw data downloading...")
    os.makedirs(save_dir, exist_ok=True)
    fs.get(rpath=gcs_file_path, lpath=data_dir, recursive=True)
    cases = sorted([c.split('/')[-1] for c in glob(f"{data_dir}/IXI*")])

    for case in cases:
        save_case_dir = os.path.join(save_dir, case)
        os.makedirs(save_case_dir, exist_ok=True)
        process_case(case, data_dir, save_dir)
    return save_dir
