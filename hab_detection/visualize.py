import os
import re
from sklearn.metrics import ConfusionMatrixDisplay
import zipfile
import seaborn as sns
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib
import numpy as np
import torch
import pprint
from .constants import (
    device,
    cyan_colormap,
    LOG_NAME,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    MODEL_SAVE_BASE_FOLDER,
    FULL_IMAGE_BASE_FOLDER,
)

from .metrics import get_metric_tracker, get_model_performance
from .helpers import log, set_config
from .dataset import get_image_dataset, transform_input
from .model import load_model


def save_plot(image_save_folder, filename):
    plt.savefig(f"{image_save_folder}/{filename}.png")


def normalize_sen2(red, green, blue):
    def normalize(arr):
        """Function to normalize an input array to 0-1"""
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)

    img = np.dstack((normalize(red), normalize(green), normalize(blue)))

    # Increase contrast
    pixvals = img
    minval = np.percentile(pixvals, 5)
    maxval = np.percentile(pixvals, 95)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 1
    Image.fromarray(pixvals.astype(np.uint8))
    return pixvals


def visualize_patch(
    model,
    dataset,
    class_designation,
    image_save_folder,
    tile_base_name,
    zip_path,
):
    z = zipfile.ZipFile(zip_path, mode="r")
    sen2_np = np.load(z.open(f"{tile_base_name}_sen2.npy"))
    cyan_np = np.load(z.open(f"{tile_base_name}_cyan.npy"))
    z.close()

    return visualize_image(
        model,
        dataset,
        class_designation,
        image_save_folder,
        tile_base_name,
        sen2_np,
        cyan_np,
        is_patch=True,
    )


def visualize_full_image(
    model,
    dataset,
    class_designation,
    image_save_folder,
    image_name,
):
    input_path = f"{FULL_IMAGE_BASE_FOLDER}/{image_name}/sen2.npy"
    label_path = f"{FULL_IMAGE_BASE_FOLDER}/{image_name}/cyan.npy"
    sen2_np = np.load(input_path).astype(np.float32)
    cyan_np = np.load(label_path)
    pred_np = np.empty(cyan_np.shape)

    batch = np.empty((0, 12, 64, 64), dtype=sen2_np.dtype)
    target_indices = []
    x_len = sen2_np.shape[1]
    y_len = sen2_np.shape[2]
    print(sen2_np.shape)
    for x in range(0, x_len, 64):
        for y in range(0, y_len, 64):
            used_x = x
            used_y = y
            if used_x + 64 > x_len:
                used_x = x_len - 64
            if used_y + 64 > y_len:
                used_y = y_len - 64

            print(used_x, used_y)
            batch = np.concatenate(
                (
                    batch,
                    np.expand_dims(sen2_np[:, used_x : used_x + 64, used_y : used_y + 64], 0),
                ),
                axis=0,
            )
            target_indices.append(
                {
                    "x_target": x,
                    "x_offset": x - used_x,
                    "y_target": y,
                    "y_offset": y - used_y,
                }
            )
            if batch.shape[0] == 32 or (x_len - 64 <= x and y_len - 64 <= y):
                with torch.no_grad():
                    model.eval()
                    transformed_batch = transform_input(
                        torch.from_numpy(batch.astype(np.float32) / 10000),
                    ).to(device, dtype=torch.float)
                    pred = model.predict(transformed_batch)  # make prediction
                    pred = pred.cpu().detach()
                    pred = np.squeeze(
                        torch.argmax(pred, dim=1, keepdim=False).cpu().numpy()
                    )
                    print(pred.shape)
                    for i, target_index in enumerate(target_indices):
                        x_target = target_index["x_target"]
                        x_offset = target_index["x_offset"]
                        y_target = target_indey["y_target"]
                        y_offset = target_indey["y_offset"]
                        print(
                            "TARGET SHAPE",
                            pred_np[
                                :,
                                x_target : x_target + 64 - x_offset,
                                y_target : y_target + 64 - x_offset,
                            ].shape,
                        )
                        print("VALUE SHAPE", pred[i, :, x_offset:, y_offset:])
                        pred_np[
                            :,
                            x_target : x_target + 64 - x_offset,
                            y_target : y_target + 64 - x_offset,
                        ] = pred[i, :, x_offset:, y_offset:]

                target_indices = []
                batch = np.empty((0, 12, 64, 64), dtype=sen2_np.dtype)
                return
    return

    tracker = get_metric_tracker(class_designation)
    print(pred_np.shape)
    print(cyan_np.shape)
    """
    tracker.update(
        pred_np,
        cyan_np,
    )
    """
    transformed_sen2 = transform_input(
        torch.from_numpy(sen2_np.astype(np.float32) / 10000),
    )

    transformed_sen2 = transformed_sen2.to(device, dtype=torch.float)
    transformed_sen2_batch = torch.unsqueeze(transformed_sen2, axis=0)

    # TODO RETURN TWO AFTER IMAGE TESTING
    pred = model.predict(transformed_sen2_batch)  # make prediction

    label = torch.unsqueeze(
        dataset.transform_label(torch.from_numpy(cyan_reshaped).int()), 0
    ).to(device)

    pred = pred.cpu().detach()

    pred = np.squeeze(torch.argmax(pred, dim=1, keepdim=False).cpu().numpy())
    pred_masked = np.where(
        cyan_reshaped > 253, 255, np.array(class_designation)[pred] - 1
    )
    log(
        f"MulticlassAccuracy for {image_name}: {tracker.compute_all()['MulticlassAccuracy'][0]}"
    )

    return visualize_image(
        class_designation,
        image_save_folder,
        image_name,
        sen2_np,
        cyan_np,
        pred_np,
    )


def visualize_image(
    class_designation,
    image_save_folder,
    image_name,
    sen2_np,
    cyan_np,
    pred_np,
):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    height = sen2_np.shape[1]
    width = sen2_np.shape[2]
    ycrop = height % 8
    xcrop = width % 8
    sen2_np = sen2_np[:, 0 : height - ycrop, 0 : width - xcrop]
    sen2_img = normalize_sen2(sen2_np[1, :, :], sen2_np[2, :, :], sen2_np[3, :, :])
    ax = axs[1, 0]
    ax.set_title("Actual image")
    ax.imshow(sen2_img)
    ax.axis("off")

    cyan_reshaped = cyan_np.reshape(cyan_np.shape[1], cyan_np.shape[2])[
        0 : height - ycrop, 0 : width - xcrop
    ]
    ax = axs[0, 0]
    ax.set_title("Actual HAB Index")
    ax.imshow(cyan_colormap[cyan_reshaped])
    ax.axis("off")

    custom_colormap = np.copy(cyan_colormap)
    prev_val = 0
    used = list(range(len(cyan_colormap)))
    for i, c in enumerate(class_designation):
        cur_color = cyan_colormap[c - 1 if i != 0 else 0]
        for j in range(c - prev_val):
            custom_colormap[prev_val + j] = cur_color
        prev_val = c

    cyan_image = custom_colormap[cyan_reshaped]
    ax = axs[0, 1]
    ax.set_title("Actual HAB Class")
    ax.imshow(cyan_image)
    ax.axis("off")

    ax = axs[1, 1]
    ax.set_title("Prediction HAB Class")
    ax.imshow(custom_colormap[pred_np])
    ax.axis("off")

    save_plot(image_save_folder, image_name)
    # log(
    #    f"MulticlassAccuracy for {image_name}: \n\n{pprint.pformat(tracker.compute_all())}"
    # )


def visualize(
    experiment_name,
    class_designation,
    model_architecture,
    model_file,
    dataset_type="test",
):
    # Set the config to print to stdout
    set_config("test")

    model_save_folder = f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}"
    image_save_folder = f"{model_save_folder}/visualize/{dataset_type}"
    log_file = f"{model_save_folder}/{LOG_NAME}"

    os.makedirs(image_save_folder, exist_ok=True)
    log(f"Loading the model")

    model = load_model(
        model_architecture, model_file, model_save_folder, class_designation
    )

    log(f"Loading the dataset")
    if dataset_type == "test":
        dataset = get_image_dataset(ZIP_PATH_TEST, class_designation, randomize=False)
    elif dataset_type == "train":
        dataset = get_image_dataset(ZIP_PATH_TRAIN, class_designation, randomize=False)
    else:
        raise Exception(f"Unknown dataset type {dataset_type}")
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    """
    print("Calculating batch vs individual performance")

    with torch.no_grad():
        model.eval()
        counter = 0
        for batch_idx, (inputs, labels, _, _) in enumerate(loader):
            tracker = get_metric_tracker(class_designation)
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device)
            # TODO PUT BACK AFTER VUISUALIZATION TESTING
            preds_old = model(inputs)  # make prediction
            preds = model.predict(inputs)  # make prediction

            tracker.update(preds, labels)
            acc = tracker.compute_all()["MulticlassAccuracy"][0]

            acc_total = 0
            for image_index in range(inputs.shape[0]):
                itracker = get_metric_tracker(class_designation)
                inp = inputs[image_index, :, :, :]
                label = labels[image_index, :, :]
                pred = model.predict(torch.unsqueeze(inp, 0))  # make prediction
                itracker.update(pred, torch.unsqueeze(label, 0))
                iacc = itracker.compute_all()["MulticlassAccuracy"][0]
                acc_total += iacc

            print(f"Batch: {acc} ---- Avg individual: {acc_total/(image_index + 1)}")
            counter += 1
            if counter >= 2:
                break
    """

    log("Visualizing full images.")
    visualize_full_image(
        model,
        dataset,
        class_designation,
        image_save_folder,
        "winnebago",
    )
    return
    visualize_full_image(
        model,
        dataset,
        class_designation,
        image_save_folder,
        "greenbay",
    )
    visualize_full_image(
        model,
        dataset,
        class_designation,
        image_save_folder,
        "erie",
    )
    """
    visualize_patch(
        model,
        dataset,
        class_designation,
        image_save_folder,
        "winnebago_2019_7_25_x32_y1600_64x64_28",
        ZIP_PATH_TRAIN,
    )
    visualize_patch(
        model,
        dataset,
        class_designation,
        image_save_folder,
        "winnebago_2019_7_25_x864_y1056_64x64_788",
        ZIP_PATH_TRAIN,
    )
    """
    log("Done visualizing full images.")

    log("Generating loss plot.")
    try:
        test_loss = []
        train_loss = []
        cur_epoch = None
        with open(log_file, "r") as f:
            for line in f.readlines():
                result = re.search(r"^.*Epoch (\d*) ([a-z]*) loss: (\d*[.]?\d*)", line)
                if result is not None:
                    epoch, t, loss = result.groups()
                    if cur_epoch is None:
                        cur_epoch = epoch
                    else:
                        # Assert that train and test don't get misaligned
                        assert cur_epoch == epoch
                        cur_epoch = None
                    if t == "test":
                        test_loss.append(float(loss))
                    elif t == "train":
                        train_loss.append(float(loss))
                    else:
                        raise Exception(f"Found unknown type {t} in log")

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(test_loss, color="b", label="Test")
        ax.plot(train_loss, color="r", label="Train")
        ax.set(xlabel="Epoch", ylabel="Loss")

        plt.legend()
        plt.title("Loss across training epochs")

        # plt.show()
        save_plot(image_save_folder, "loss")
    except FileNotFoundError as e:
        log("Log file not found. Skipping the loss plot.")

    _, metrics, hist_2d = get_model_performance(
        model,
        loader,
        class_designation,
        num_batches=-1,
        calculate_2d_hist=True,
    )
    log(f"\n{pprint.pformat(metrics)}")

    """ Confusion Matrix """
    cm = np.squeeze(metrics["MulticlassConfusionMatrix"].cpu().numpy())
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    vmin = np.min(cmn)
    vmax = np.max(cmn)
    off_diag_mask = np.eye(*cmn.shape, dtype=bool)

    fig = plt.figure()
    gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs0[1], hspace=0
    )

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    # cax2 = fig.add_subplot(gs00[1])

    class_names = [
        f"{0 if i == 0 else class_designation[i-1]} - {a - 1}"
        for i, a in enumerate(class_designation)
    ]
    """
    sns.heatmap(
        cmn,
        annot=True,
        mask=~off_diag_mask,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_ax=cax2,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    """
    ax = sns.heatmap(
        cmn,
        annot=True,
        # mask=off_diag_mask,
        cmap="OrRd",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        cbar_ax=cax1,
        # cbar_kws=dict(ticks=[]),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    """
    cf_disp = ConfusionMatrixDisplay(cm)
    cf_disp.plot()
    """
    save_plot(image_save_folder, "confusion_matrix")

    if hist_2d is not None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        sums = hist_2d.astype("float").sum(axis=0) + 1
        print("SUMS SHAPE", sums.shape)
        print("SUMS", sums)
        rectangles = {}
        ranges = [
            (
                -20 if i == 0 else class_designation[i - 1],
                class_designation[i],
            )
            for i in range(len(class_designation))
        ]

        for i in range(len(class_designation)):
            floor = ranges[i][0]
            ceil = ranges[i][1]
            color = cyan_colormap[ceil - 1 if floor != 0 else 0] / 255

            rectangles[f"{floor} - {ceil -1}" if i != 0 else "0"] = mpatch.Rectangle(
                (floor, 1.02),
                ceil - floor,
                0.05,
                color=color,
            )

            normalized = hist_2d[i] / sums
            axs.plot(
                range(1, 254),
                normalized[1:],
                label=f"Class {i + 1}",
                color=color,
                alpha=0.3,
                linewidth=2.0,
            )
            # Plot thicker line inside of range where it is correct
            if i != 0:
                axs.plot(
                    range(floor, min(ceil + 1, len(normalized))),
                    normalized[floor : min(ceil + 1, len(normalized))],
                    color=color,
                    linewidth=2.0,
                )
            plt.plot(
                [-20, 0.5],
                [normalized[0], normalized[0]],
                linewidth=2.0,
                alpha=0.3 if i != 0 else 1.0,
                color=color,
            )
        for i in range(len(class_designation)):
            plt.axvline(x=class_designation[i], color="black", alpha=1.0)

        # Plot rectangles
        for r in rectangles:
            rect = rectangles[r]
            axs.add_artist(rect)
            rx, ry = rect.get_xy()
            cx = rx + rect.get_width() / 2.0
            cy = ry + rect.get_height() / 2.0

            axs.annotate(
                r,
                (cx, cy),
                color="white",
                weight="bold",
                fontsize=10,
                ha="center",
                va="center",
            )

        plt.autoscale(enable=True, axis="x", tight=True)
        axs.set_ylim(0.0, 1.07)
        axs.set_xlim(-20, 253)
        plt.xticks([-10, 1, 100, 150, 200, 253], [0, 1, 100, 150, 200, 253])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0])

        axs.set_title("Classification of Pixels vs Actual HAB Index")
        plt.xlabel("Actual HAB Index")
        plt.ylabel("Fraction Classified")

        save_plot(image_save_folder, "class_preds")
