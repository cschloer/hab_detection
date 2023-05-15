from .constants import dataset_mean, dataset_std
from .helpers import log

import time
import psutil
import os
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

transform_input = transforms.Compose([transforms.Normalize(dataset_mean, dataset_std)])


def get_data(zip_path, use_unzipped=False):
    imgs = []
    labels = []
    namelist = []
    if use_unzipped:
        zip_path = zip_path[:-4]
        log(f"ZIP PATH {zip_path}")
        namelist = set(os.listdir(zip_path))
        log(f"Numfiles: {len(namelist)}")
    else:
        zip = zipfile.ZipFile(zip_path, mode="r")
        namelist = set(zip.namelist())
        zip.close()


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

            imgs.append(f)
            labels.append(label_filename)

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
        in_memory=False,
        fold_list=None,
        use_unzipped=False,
        in_memory_prefill=True,
    ):
        super().__init__()
        assert len(imgs) == len(labels)
        self.imgs = imgs
        self.labels = labels
        self.zip_path = zip_path
        self.class_designation = class_designation
        self.randomize = randomize
        self.do_transform = transform
        self.zip = None
        self.in_memory = in_memory
        self.use_unzipped = use_unzipped
        if in_memory:
            self.cache = [None] * len(self.imgs)
            if in_memory_prefill:
                total_size = len(self.imgs)
                log(f"Loading a dataset with {total_size} images into memory")
                for idx in range(total_size):
                    self._get_image(idx)
                    if (idx + 1) % int(total_size / 10) == 0:
                        log(
                            f"Loaded {idx + 1} images of {total_size} -- using {round(psutil.Process(os.getpid()).memory_info().rss / (1<<30), 2)} GB"
                        )
                self.close_zip()
                log(
                    f"After close zip -- using {round(psutil.Process(os.getpid()).memory_info().rss / (1<<30), 2)} GB"
                )

        # Initialize lists for kfold
        self.fold_list = fold_list
        if self.fold_list is not None:
            assert len(self.fold_list) == len(imgs)
            self.backup_imgs = imgs
            self.backup_labels = labels
            self.backup_cache = self.cache
            log(
                f"After backup -- using {round(psutil.Process(os.getpid()).memory_info().rss / (1<<30), 2)} GB"
            )

    def __len__(self):
        if self.fold_list is not None:
            return len([fold for fold in self.fold_list if fold != self.fold])
        return len(self.imgs)

    def set_fold(self, fold, is_train):
        if self.fold_list is not None:
            self.fold = fold
            bool_list = [
                (is_train and fold_idx != self.fold)
                or (not is_train and fold_idx == self.fold)
                for fold_idx in self.fold_list
            ]
            self.imgs = [img for i, img in enumerate(self.backup_imgs) if bool_list[i]]
            self.labels = [
                label for i, label in enumerate(self.backup_labels) if bool_list[i]
            ]
            self.cache = [
                cache_item
                for i, cache_item in enumerate(self.backup_cache)
                if self.fold_list[i] != self.fold
            ]
            if is_train:
                log(f"Fold set, new size: {len(self.imgs)}")

    def _get_image(self, idx):
        if self.in_memory and self.cache[idx] is not None:
            return self.cache[idx]
        filename = self.imgs[idx]
        label_filename = self.labels[idx]
        image = None
        label = None
        if self.use_unzipped:
            image = np.load(f"{self.zip_path}/{filename}")
            label = np.load(f"{self.zip_path}/{label_filename}")

        else:
            if self.zip == None:
                self.open_zip()
            image = np.load(self.zip.open(filename))
            label = np.load(self.zip.open(label_filename))
        if image is None:
            raise FileNotFoundError(filename)
        if label is None:
            raise FileNotFoundError(label_filename)
        if self.in_memory:
            self.cache[idx] = (
                image,
                label,
            )
        return image, label

    def get_untransformed_image(self, idx):
        return self._get_image(idx)

    def get_image_filename(self, idx):
        return self.imgs[idx]

    def close_zip(self):
        if self.zip and not self.use_unzipped:
            self.zip.close()
            self.zip = None

    def open_zip(self):
        if not self.use_unzipped:
            self.close_zip()
            self.zip = zipfile.ZipFile(self.zip_path, mode="r")

    def mask_label(self, label):
        return torch.where(label >= 254, -1, label)

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
            label = torch.where((label >= floor) & (label < ceil), i, label)
            floor = ceil

        # label = torch.from_numpy(label)
        # F.one_hot(label, num_classes=len(class_designation))

        label = torch.squeeze(label.long())
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

        """
        # Random crop
        x_offset = random.randrange(16)
        y_offset = random.randrange(16)
        cropped_size = 64 - max([random.randrange(16), x_offset, y_offset])
        image = TF.resized_crop(
            image, y_offset, x_offset, cropped_size, cropped_size, 64
        )
        label = TF.resized_crop(
            label, y_offset, x_offset, cropped_size, cropped_size, 64
        )

        # Gaussian Blur
        sigma = np.random.uniform(0.1, 2.0)
        kernel_size = 7
        image = TF.gaussian_blur(image, kernel_size, sigma)
        """

        return image, label

    def __getitem__(self, idx):
        raw_image, raw_label = self._get_image(idx)

        # We divide to make the numbers manageable for calculating mean and std and thus normalizing
        # 10,000 is the accepted number to get a brightness level
        image = raw_image.astype(np.float32) / 10000
        # augmentations
        if self.do_transform:
            image = transform_input(torch.from_numpy(image))
            label = torch.from_numpy(raw_label).int()
            if self.randomize:
                image, label = self.random_transform(image, label)
            label = self.transform_label(label)
        else:
            label = raw_label

        return (
            image,
            label,
            raw_image.astype(np.float32),
            torch.squeeze(torch.from_numpy(raw_label.astype(np.int_))),
        )


def get_image_dataset(
    zip_path,
    class_designation,
    randomize=False,
    transform=True,
    # A fraction of the total dataset to use
    subset=None,
    in_memory=False,
    use_unzipped=False
):
    imgs, labels, zip_path = get_data(zip_path, use_unzipped=use_unzipped)
    if subset is not None:
        random.seed("subset")
        combined = list(zip(imgs, labels))
        random.shuffle(combined)
        imgs[:], labels[:] = zip(*combined)
        end_index = int(subset * len(imgs))
        imgs = imgs[:end_index]
        labels = labels[:end_index]
    image_dataset = ImageData(
        imgs,
        labels,
        zip_path,
        class_designation,
        randomize=randomize,
        transform=transform,
        in_memory=in_memory,
        use_unzipped=use_unzipped,
    )
    return image_dataset
