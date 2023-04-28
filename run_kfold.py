from hab_detection.train import train
from hab_detection.helpers import log, set_config
from hab_detection.dataset import get_data, ImageData
from hab_detection.metrics import get_model_performance
from hab_detection.constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
)
import numpy as np
import pprint
import json
import os
import random
from torch.utils.data import DataLoader

NUM_FOLDS = 5
SUBSET_SIZE = 50000

with open("experiments.json", "r") as f:
    experiments = json.load(f)
e = experiments["experiment25"]

experiment_name = "kfold_test"
model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
os.makedirs(model_save_folder, exist_ok=True)
set_config(experiment_name)
log(f"Starting kfold experiment")


random.seed("kfold_test")
imgs, labels, zip_path = get_data(ZIP_PATH_TRAIN)
combined = list(zip(imgs, labels))
random.shuffle(combined)
imgs[:], labels[:] = zip(*combined)
imgs = imgs[:SUBSET_SIZE]
label = labels[:SUBSET_SIZE]

region_folds = {}
counter = 0
printed = False
fold_indices = [None] * len(imgs)
for i, img in enumerate(imgs):
    match = re.findall(
        "([a-z0-9_]*)_\d\d\d\d_.*_sen2.npy",
        img,
        re.IGNORECASE,
    )
    if match:
        (region,) = match[0]
        if not printed:
            print(region)
            printed = True
        if region not in region_folds:
            region_folds[region] = counter
            counter = (counter + 1) % NUM_FOLDS
        fold_indices[i] = region_folds[region]
    else:
        raise Exception("Found a non match", img)

for fold in range(NUM_FOLDS):
    log(
        f"Fold {fold + 1}: {len([fold_index for fold_index in fold_indices if fold_index != fold])} images"
    )

results = {}
train_dataset = ImageData(
    imgs,
    labels,
    zip_path,
    e["class_designation"],
    randomize=True,
    transform=True,
    in_memory=True,
)
test_dataset = ImageData(
    imgs,
    labels,
    zip_path,
    e["class_designation"],
    randomize=False,
    transform=True,
    in_memory=True,
)


for model_arc in [
    "deeplabv3_mobilenet_v3_large#no_replace_batchnorm",
    "lraspp_mobilenet_v3_large#no_replace_batchnorm",
]:
    for learning_rate in [0.00001, 0.0001, 0.001]:
        for batch_size in [16, 32, 64]:
            for weight_decay in [0.02, 0.002]:
                log(
                    f"Testing model {model_arc} with learning rate {learning_rate}, batch size {batch_size}, and weight decay {weight_decay}."
                )
                losses = []
                for fold in range(NUM_FOLDS):
                    log(f"Starting fold {fold + 1}")
                    # Set images to fold, but keep cache
                    train_dataset.imgs = [
                        imgs[fold_index]
                        for fold_index in fold_indices
                        if fold_index != fold
                    ]
                    train_dataset.labels = [
                        labels[fold_index]
                        for fold_index in fold_indices
                        if fold_index != fold
                    ]
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True,
                    )
                    test_dataset.imgs = [
                        imgs[fold_index]
                        for fold_index in fold_indices
                        if fold_index == fold
                    ]
                    test_dataset.labels = [
                        labels[fold_index]
                        for fold_index in fold_indices
                        if fold_index == fold
                    ]
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        drop_last=True,
                    )
                    model = train(
                        train_loader,
                        test_loader,
                        experiment_name,
                        batch_size,
                        e["class_designation"],
                        e["class_weights"],
                        model_arc,
                        True,
                        learning_rate,
                        model_file=None,
                        epoch_start=0,
                        weight_decay=weight_decay,
                        track_statistics=False,
                        log_progress=False,
                        save_progress=False,
                        epoch_limit=20,
                    )
                    test_loss, _, _ = get_model_performance(
                        model,
                        test_loader,
                        e["class_designation"],
                        class_weights=e["class_weights"],
                        calculate_statistics=False,
                    )
                    losses.append(test_loss)
                    log(f"Finished fold {fold + 1} with test loss: {test_loss}")
                average_loss = np.average(losses)
                log(
                    f"Finished model {model_arc}, learning rate {learning_rate}, batch size {batch_size}, weight decay {weight_decay}.\nAverage loss: {average_loss}"
                )
                key = f"{model_arc} --- LR: {learning_rate} --- BS: {batch_size} -- WD: {weight_decay}"
                results[key] = average_loss
                log("---------------\n")


sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
log(f"\n{pprint.pformat(sorted_results)}")
