import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomVerticalFlip, RandomAffine


class RandomGeneratorIXI(object):
    def __init__(self, scale=None, flip=False, n_contrast=4):
        self.scale = scale
        self.flip = flip
        self.n_contrast = n_contrast

    def __call__(self, data):
        n_contrast = self.n_contrast
        image = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
        if self.flip:
            image = RandomVerticalFlip(p=0.5)(image)
        if self.scale:
            image = RandomAffine(0, scale=self.scale, p=1)(image)
        image = image.detach()
        # load images
        output = [image[:, i, :, :] for i in range(n_contrast)]
        return output


class IXI_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.data_dir = os.path.join(base_dir, split)
        data_list = []
        cases = glob.glob(f"{self.data_dir}/IXI*")
        for case in cases:
            files = glob.glob(f'{case}/*.npy')
            data_list += files
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        if self.transform:
            data = self.transform(data)
        return data


class IXISingleDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform  # Using the transform parameter from torch to preprocess data.
        self.data_dir = base_dir
        data_list = []
        files = glob.glob(f'{self.data_dir}/*.npy')  # Get a list of all .npy files in the specified directory.
        data_list += files
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)  # Return the number of items in the dataset.

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])  # Load data from a specific .npy file.
        if self.transform:
            data = self.transform(data)  # Apply the specified transformation to the data if provided.
        return data  # Return the processed data item.
