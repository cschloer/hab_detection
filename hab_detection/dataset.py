from constants import dataset_mean, dataset_std

import torchvision.transforms as transforms
import zipfile
import re
from torch.utils.data import Dataset
import numpy as np


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

            imgs.append(f)
            labels.append(label_filename)

    zip.close()
    return imgs, labels, zip_path


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(dataset_mean, dataset_std)]
)


class ImageData(Dataset):
    def __init__(
        self,
        imgs,
        labels,
        zip_path,
    ):
        super().__init__()
        self.imgs = imgs
        self.labels = labels
        self.zip_path = zip_path
        self.zip = None
        self.open_zip()

    def __len__(self):
        return len(self.imgs)

    def _get_image(self, idx):
        filename = self.imgs[idx]
        try:
            image = np.load(self.zip.open(filename))
        except Exception as e:
            print("FAILED", filename)
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

    def __getitem__(self, idx):
        image, label = self._get_image(idx)
        image = image.transpose(1, 2, 0)
        image = image[:, :, 0:12]

        # We divide to make the numbers manageable for calculating mean and std and thus normalizing
        # 10,000 is the accepted number to get a brightness level
        image = image.astype(np.float32) / 10000
        # augmentations
        image = transform(image)

        return image, label, idx


def get_image_dataset(zip_path):
    imgs, labels, zip_path = get_data(zip_path)
    image_dataset = ImageData(imgs, labels, zip_path)
    return image_dataset
