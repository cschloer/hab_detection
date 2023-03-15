from .constants import dataset_mean, dataset_std
from .helpers import log

import torchvision.transforms as transforms
import math
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import zipfile
import re
from torch.utils.data import Dataset
import numpy as np
import time


def get_data(zip_path):
    imgs = []
    labels = []
    zip = zipfile.ZipFile(zip_path, mode="r")
    namelist = set(zip.namelist())

    for f in namelist:
        match = re.findall(
            "([a-z0-9_]*)_(\d\d\d\d)_(\d*)_(\d*)_x(\d*)_y(\d*)_(\d*)x(\d*)_([a-z0-9]*)_sen2.npy",
            f,
            re.IGNORECASE,
        )
        if match:
            region, year, month, day, x_start, y_start, tile_size, _, id = match[0]

            label_filename = f"{region}_{year}_{month}_{day}_x{x_start}_y{y_start}_{tile_size}x{tile_size}_{id}_cyan.npy"
            if not label_filename in namelist:
                raise Exception(
                    f'Corresponding label file doesn\'t exist: "{label_filename}"'
                )

            if f.startswith("winnebago_2019_7_25_"):
                imgs.append(f)
                labels.append(label_filename)

    print("LEN", len(imgs))
    zip.close()
    return imgs, labels, zip_path


class ImageData(Dataset):
    def __init__(
        self,
        imgs,
        labels,
        zip_path,
        class_designation,
        randomize=False,
        transform=True,
    ):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.zip_path = zip_path
        self.class_designation = class_designation
        self.randomize = randomize
        self.do_transform = transform

        self.zip = None
        self.transform_input = transforms.Compose(
            [transforms.Normalize(dataset_mean, dataset_std)]
        )

    def __len__(self):
        return len(self.imgs)

    def _get_image(self, idx):
        if self.zip == None:
            self.open_zip()

        filename = self.imgs[idx]
        try:
            image = np.load(self.zip.open(filename))
        except Exception as e:
            log(f"FAILED: {filename}")
            raise e
        if image is None:
            raise FileNotFoundError(filename)

        label_filename = self.labels[idx]
        label = np.load(self.zip.open(label_filename))
        if label is None:
            raise FileNotFoundError(label_filename)
        return image, label

    def get_untransformed_image(self, idx):
        return self._get_image(idx)

    def get_image_filename(self, idx):
        return self.imgs[idx]

    def close_zip(self):
        if self.zip:
            self.zip.close()
            self.zip = None

    def open_zip(self):
        self.close_zip()
        self.zip = zipfile.ZipFile(self.zip_path, mode="r")

    def mask_label(self, label):
        return np.where((label >= 254), -1, label)

    def transform_label(self, label):
        # First set no data values to -1
        label = self.mask_label(label)
        if self.class_designation is None:
            # It's a regression problem, no need to transform to class problem
            return label
            # return torch.from_numpy(label)
        if self.class_designation[-1] != 254:
            raise Exception("The last value of the class_designation must be 254.")
        floor = 0
        for i, ceil in enumerate(self.class_designation):
            label = np.where((label >= floor) & (label < ceil), i, label)
            floor = ceil

        # label = torch.from_numpy(label)
        # F.one_hot(label, num_classes=len(class_designation))

        label = torch.squeeze(torch.from_numpy(label.astype(np.int_)))
        return label

    def random_transform(self, image, label):

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Random crop
        x_offset = random.randrange(32)
        y_offset = random.randrange(32)
        image = TF.crop(image, y_offset, x_offset, 32, 32)
        label = TF.crop(label, y_offset, x_offset, 32, 32)

        # Gaussian Blur
        sigma = np.random.uniform(0.1, 2.0)
        kernel_size = 7
        image = TF.gaussian_blur(image, kernel_size, sigma)

        return image, label

    def __getitem__(self, idx):
        raw_image, raw_label = self._get_image(idx)

        # We divide to make the numbers manageable for calculating mean and std and thus normalizing
        # 10,000 is the accepted number to get a brightness level
        image = raw_image.astype(np.float32) / 10000
        # augmentations
        if self.do_transform:
            image = self.transform_input(torch.from_numpy(image))
            label = self.transform_label(raw_label)
        else:
            label = raw_label

        if self.randomize:
            image, label = self.random_transform(image, label)

        return (
            image,
            label,
            raw_image.astype(np.int16),
            torch.squeeze(torch.from_numpy(raw_label.astype(np.int_))),
        )


def get_image_dataset(zip_path, class_designation, randomize=False, transform=True):
    imgs, labels, zip_path = get_data(zip_path)
    image_dataset = ImageData(
        imgs,
        labels,
        zip_path,
        class_designation,
        randomize=randomize,
        transform=transform,
    )
    return image_dataset
