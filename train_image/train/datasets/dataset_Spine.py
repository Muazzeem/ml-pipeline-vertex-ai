import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from kornia.augmentation import RandomVerticalFlip, RandomAffine
from skimage.transform import resize

# et = patient_label == 4
# et_present = 1 if np.sum(et) >= 1 else 0
# tc = np.logical_or(patient_label == 4, patient_label == 1)
# wt = np.logical_or(tc, patient_label == 2)
# patient_label = np.stack([et, tc, wt])


class RandomGeneratorSpine(object):
    def __init__(self, scale=None, flip=False, n_contrast=3):
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


class Spine_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None, random_crop=False, crop_size=None):
        self.transform = transform  # using transform in torch!
        self.data_dir = os.path.join(base_dir, split)
        data_list = []
        cases = glob.glob(f"{self.data_dir}/*")
        for case in cases:
            files = glob.glob(f'{case}/*.npy')
            data_list += files
        self.data_list = data_list
        self.random_crop = random_crop
        self.crop_size = crop_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        if self.crop_size is not None:
            min_shape = min(data.shape[1], data.shape[2])
            if min_shape < self.crop_size:
                data = data.transpose(1, 2, 0)
                if data.shape[1] == min_shape:
                    data = resize(data, (self.crop_size, int(self.crop_size*data.shape[2]/data.shape[1])))
                else:
                    data = resize(data, (int(self.crop_size*data.shape[1]/data.shape[2]), self.crop_size))
                data = data.transpose(2, 0, 1)
            if self.random_crop:
                h, w = data.shape[1:]
                x_top = np.random.randint(0, h - self.crop_size) if h > self.crop_size else 0
                y_top = np.random.randint(0, w - self.crop_size) if w > self.crop_size else 0
            else:
                x_top = 0
                y_top = 0
            data = data[:, x_top:x_top + self.crop_size, y_top:y_top + self.crop_size]
        if self.transform:
            data = self.transform(data)
        return data
