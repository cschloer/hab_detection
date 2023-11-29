# A script meant to find the patches with high HAB and duplicate them for better dataset distribution
import os
import re
import numpy as np

data_path = "/ssd/hab/data/dataset_train_structured"


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


def load(file):
    with open(file, "rb") as f:
        header = f.read(128)
        if not header:
            return None
        descr = str(header[19:25], "utf-8").replace("'", "").replace(" ", "")
        shape = tuple(
            int(num)
            for num in str(header[60:120], "utf-8")
            .replace(", }", "")
            .replace("(", "")
            .replace(")", "")
            .split(",")
        )
        datasize = np.lib.format.descr_to_dtype(descr).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape, dtype=descr, buffer=f.read(datasize))


def get_useful_images():
    useful_images = []
    count = 0
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

                    label = load(label_path)
                    occurances = np.count_nonzero(label > 220)
                    count += 1
                    if occurances > 1000:
                        useful_images.append(
                            (
                                occurances,
                                label_path,
                                feature_path,
                            )
                        )
                        if len(useful_images) % 1000 == 0:
                            print(
                                f"Found {len(useful_images)} of {count} useful images..."
                            )

    return useful_images


useful_images = get_useful_images()

count = 0
for useful_image in useful_images:
    occurances, label_path, feature_path = useful_image
    new_path = f"{data_path}/synthetic/{label_path[len(data_path) + 1:-1 * len(os.path.basename(label_path))]}"
    label = load(label_path)
    feature = load(feature_path)

    num_copies = 100
    if occurances > 2000:
        num_copies = 500
        if occurances > 3000:
            num_copies = 1000
    num_copies = 1

    for rotation in range(4):
        for i in range(num_copies):
            count += 1
        continue
        np.rot90(
            label,
            axes=(
                1,
                2,
            ),
        )
        np.rot90(
            feature,
            axes=(
                1,
                2,
            ),
        )
print(f"Will save {count} new images")