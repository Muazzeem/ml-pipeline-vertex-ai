import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from kornia.augmentation import RandomVerticalFlip, RandomAffine

# et = patient_label == 4
# et_present = 1 if np.sum(et) >= 1 else 0
# tc = np.logical_or(patient_label == 4, patient_label == 1)
# wt = np.logical_or(tc, patient_label == 2)
# patient_label = np.stack([et, tc, wt])


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
