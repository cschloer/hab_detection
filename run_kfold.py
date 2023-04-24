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
import pprint
import json
import os
import random
from torch.utils.data import DataLoader

with open("experiments.json", "r") as f:
    experiments = json.load(f)
e = experiments["experiment25"]

experiment_name = "kfold_test"
model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
os.makedirs(model_save_folder, exist_ok=True)
set_config(experiment_name)


random.seed("kfold_test")
imgs, labels, zip_path = get_data(zip_path)
combined = list(zip(images, labels))
random.shuffle(combined)
imgs[:], labels[:] = zip(*combined)
SUBSET_SIZE = 50000
FOLD_SIZE = 10000
SUBSET_SIZE = 500
FOLD_SIZE = 100
imgs = imgs[:SUBSET_SIZE]
label = labels[:SUBSET_SIZE]

results = {}


for model_arc in ["deeplabv3_mobilenet_v3_large#no_replace_batchnorm"]:
    for learning_rate in [0.000001, 0.00001, 0.0001]:
        for batch_size in [16, 32, 64]:
            log(
                f"Testing model {model_arc} with learning rate {learning_rate} and batch size {batch_size}"
            )
            losses = []
            for fold in range(SUBSET_SIZE / FOLD_SIZE):
                log(f"Starting fold {fold + 1}")
                train_dataset = ImageData(
                    imgs[: fold * FOLD_SIZE] + imgs[(fold + 1) * FOLD_SIZE :],
                    labels[: fold * FOLD_SIZE] + imgs[(fold + 1) * FOLD_SIZE :],
                    zip_path,
                    e["class_designation"],
                    randomize=True,
                    transform=True,
                )
                test_dataset = ImageData(
                    imgs[fold * FOLD_SIZE : (fold + 1) * FOLD_SIZE],
                    labels[fold * FOLD_SIZE : (fold + 1) * FOLD_SIZE],
                    zip_path,
                    e["class_designation"],
                    randomize=False,
                    transform=True,
                )
                model = train(
                    train_dataset,
                    test_dataset,
                    "kfold_testing",
                    batch_size,
                    e["class_designation"],
                    e["class_weights"],
                    model_arc,
                    True,
                    learning_rate,
                    model_file=None,
                    epoch_start=0,
                    weight_decay=0,
                    track_statistics=False,
                    log_progress=False,
                    save_progress=False,
                    epoch_limit=30,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True,
                )
                test_loss, _, _ = get_model_performance(
                    model,
                    test_loader,
                    class_designation,
                    class_weights=class_weights,
                    calculate_statistics=False,
                )
                losses.append(test_loss)
                log(f"Finished fold {fold + 1} with test loss: {test_loss}")
            average_loss = np.average(losses)
            log(
                f"Finished model {model_arc}, learning rate {learning_rate}, batch size {batch_size}.\nAverage loss: {average_loss}"
            )
            key = f"{model_arc} --- LR: {learning_rate} --- BS: {batch_size}"
            results[key] = average_loss
            log("---------------\n")

sorted_results = dict(sorted(x.items(), key=lambda item: item[1]))
log(f"\n{pprint.pformat(sorted_results)}")
