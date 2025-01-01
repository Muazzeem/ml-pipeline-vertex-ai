# Data Preparing

## Step 1: Split the dataset into train/val/test sets
Use [split_dataset.ipynb](split_dataset.ipynb). The result is saved in [data_split_brats2021.pickle](data_split_brats2021.pickle) and will be used for further processing.
## Step 2: Data preprocessing
Use [generate_dataset_brats2021_crop192x160.ipynb](generate_dataset_brats2021_crop192x160.ipynb) for data preprocessing, which will perform mean normalization, crop the images to `192x160`, and extract the slices of different contrasts. Outputs are `.npy` files that contain matrices of shape `4x192x160`. The order of channels is `T1, T1Gd, T2, FLAIR`.

## Dataset Path
The original BraTS2021 dataset is in 
`
/mnt/raid/jiang/projects/SubtleGAN/data/RSNA_ASNR_MICCAI_BraTS2021_TrainingData
`

The preprocessed dataset is in 
`
/mnt/raid/jiang/projects/SubtleGAN/data/brats2021_slices_crop192x160
`