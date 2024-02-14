from .constants import dataset_mean, dataset_std, all_dist
from .helpers import log, load

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

transform_input = transforms.Compose([transforms.Normalize(dataset_mean, dataset_std)])


def get_label_filename(feature_filename):
    match = re.findall(
        "([a-z0-9_]*)_(\d\d\d\d)_(\d*)_(\d*)_x(\d*)_y(\d*)_(\d*)x(\d*)_([a-z0-9]*)_sen2.npy",
        feature_filename,
        re.IGNORECASE,
    )
    if match:
        region, year, month, day, x_start, y_start, tile_size, _, id = match[0]

        label_filename = f"{region}_{year}_{month}_{day}_x{x_start}_y{y_start}_{tile_size}x{tile_size}_{id}_cyan.npy"
        return label_filename


def get_data(data_path, use_zip=False):
    features = []
    labels = []
    namelist = []
    if not use_zip:
        for root, dirs, files in os.walk(data_path, topdown=False):
            for name in files:
                feature_path = os.path.join(root, name)
                if os.path.isfile(feature_path):
                    feature_filename = os.path.basename(feature_path)
                    dirname = os.path.dirname(feature_path)
                    label_filename = get_label_filename(feature_filename)
                    if label_filename:
                        label_path = f"{dirname}/{label_filename}"
                        if not os.path.isfile(label_path):
                            print(
                                f'Corresponding label file doesn\'t exist: "{label_filename}"'
                            )
                            continue

                        features.append(feature_path)
                        labels.append(label_path)
                        label = load(label_path)
                        occurances = np.count_nonzero(label > 220)
                        # Increase the occurance of labels > 220
                        """
                        if occurances > 1000:
                            for i in range(4):
                                features.append(feature_path)
                                labels.append(label_path)
                        """
                        if len(labels) > 1000:
                            return features, labels, data_path if use_zip else None

    else:
        zip = zipfile.ZipFile(data_path, mode="r")
        namelist = set(zip.namelist())
        zip.close()

        for feature_filename in namelist:
            label_filename = get_label_filename(feature_filename)
            if label_filename:
                if not label_filename in namelist:
                    raise Exception(
                        f'Corresponding label file doesn\'t exist: "{label_filename}"'
                    )

                features.append(feature_filename)
                labels.append(label_filename)

    return features, labels, data_path if use_zip else None


class ImageData(Dataset):
    def __init__(
        self,
        features,
        labels,
        class_designation,
        randomize=False,
        transform=True,
        in_memory=False,
        fold_list=None,
        zip_path=None,
        in_memory_prefill=True,
    ):
        super().__init__()
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
        self.zip_path = zip_path
        self.class_designation = class_designation
        self.randomize = randomize
        self.do_transform = transform
        self.zip = None
        self.in_memory = in_memory
        self.cache = None
        if in_memory:
            self.cache = [None] * len(self.features)
            if in_memory_prefill:
                total_size = len(self.features)
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
            assert len(self.fold_list) == len(features)
            self.backup_features = features
            self.backup_labels = labels
            if in_memory:
                self.backup_cache = self.cache

    def __len__(self):
        return len(self.features)

    def set_fold(self, fold, is_train):
        if self.fold_list is not None:
            self.fold = fold
            bool_list = [
                (is_train and fold_idx != self.fold)
                or (not is_train and fold_idx == self.fold)
                for fold_idx in self.fold_list
            ]
            self.features = [
                img for i, img in enumerate(self.backup_features) if bool_list[i]
            ]
            self.labels = [
                label for i, label in enumerate(self.backup_labels) if bool_list[i]
            ]
            if self.cache:
                self.cache = [
                    cache_item
                    for i, cache_item in enumerate(self.backup_cache)
                    if self.fold_list[i] != self.fold
                ]
            log(
                f"Fold set for {'train' if is_train else 'test'}, new size: {len(self.features)}"
            )

    def _get_image(self, idx):
        # start = time.time()
        if self.in_memory and self.cache[idx] is not None:
            return self.cache[idx]
        filename = self.features[idx]
        label_filename = self.labels[idx]
        image = None
        label = None
        if not self.zip_path:
            # No zip path provided, load directly from filesystem
            image = load(filename)
            label = load(label_filename)

        else:
            if self.zip == None:
                self.open_zip()
            image = load(self.zip.open(filename))
            label = load(self.zip.open(label_filename))
        if image is None:
            raise FileNotFoundError(filename)
        if label is None:
            raise FileNotFoundError(label_filename)
        if self.in_memory:
            self.cache[idx] = (
                image,
                label,
            )
        # print(f"TOOK {time.time() - start} seconds")
        return image, label

    def get_untransformed_image(self, idx):
        return self._get_image(idx)

    def get_image_filename(self, idx):
        return self.features[idx]

    def close_zip(self):
        if self.zip and self.zip_path:
            self.zip.close()
            self.zip = None

    def open_zip(self):
        if self.zip_path:
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
    data_path,
    class_designation,
    randomize=False,
    transform=True,
    # A fraction of the total dataset to use
    subset=None,
    in_memory=False,
    use_zip=False,
):
    features, labels, zip_path = get_data(data_path, use_zip=use_zip)
    if subset is not None:
        random.seed("subset")
        combined = list(zip(features, labels))
        random.shuffle(combined)
        features[:], labels[:] = zip(*combined)
        end_index = int(subset * len(features))
        features = features[:end_index]
        labels = labels[:end_index]
    image_dataset = ImageData(
        features,
        labels,
        class_designation,
        zip_path=zip_path,
        randomize=randomize,
        transform=transform,
        in_memory=in_memory,
    )
    return image_dataset


def get_weighted_all_dist(class_designation):
    weighted_all_dist = np.ones(len(all_dist))
    for i in range(len(class_designation)):
        start = class_designation[i - 1] if i > 0 else 0
        end = class_designation[i]
        for j in range(start, end):
            weighted_all_dist[j] = all_dist[j] / (end - start)
    return weighted_all_dist
