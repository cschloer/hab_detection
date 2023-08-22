from .constants import device, MODEL_SAVE_BASE_FOLDER
from .two_head_model import create_twohead_model

from torchvision import models
from torchvision.models.segmentation import (
    fcn_resnet50,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation
import torch
import numpy as np


def convert_batchnorm2d(model):
    # Converts all BatchNorm2d to track_running_stats=False
    # due to bug/unexpected behavior in BatchNorm2D
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            num_features = child.num_features
            setattr(
                model,
                child_name,
                torch.nn.BatchNorm2d(num_features, track_running_stats=False),
            )
        else:
            convert_batchnorm2d(child)


def use_groupnorm(model):
    # Converts all BatchNorm2d to GroupNorm
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            num_features = child.num_features
            num_groups = 32
            while num_features <= num_groups or num_features % num_groups != 0:
                num_groups = int(num_groups / 2)
                if num_groups < 1:
                    # We shouldn't get here
                    assert False

            setattr(
                model,
                child_name,
                torch.nn.GroupNorm(num_groups, num_features),
            )
        else:
            use_groupnorm(child)


# A custom segmentation head that adds dropout
class SegmentationHeadDropout(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dropout=0.5,
        activation=None,
        upsampling=1,
    ):
        dropout = torch.nn.Dropout(p=dropout) if dropout else torch.nn.Identity()
        conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = (
            torch.nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else torch.nn.Identity()
        )
        activation = Activation(activation)
        super().__init__(dropout, conv2d, upsampling, activation)


def load_model(
    model_architecture,
    # model_file is None if we shouldn't load from a previous model.
    # Otherwise path starting with "/" or filename in model_folder starting without /
    model_file,
    model_folder,
    class_designation,
):
    num_classes = 2
    if class_designation is not None:
        num_classes = len(class_designation)
    if model_architecture.startswith("deeplabv3-resnet50"):
        # Load resnet50 segmentation model
        model = smp.DeepLabV3(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=12,
            classes=num_classes,
        )
        if class_designation is None:
            # Chance first layer to accept 12 input bands (12 bands)
            # model.backbone.conv1 = torch.nn.Conv2d(
            #    12, 64, kernel_size=(7, 7), stride=(2, 2)
            # )

            # Change final layer to 1 continuous output
            model.decoder.segmentation_head[4] = torch.nn.Sequential(
                torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.Sigmoid(),
            )

    elif model_architecture.startswith("deeplabv3-resnet18"):
        if class_designation is not None:
            model = smp.DeepLabV3(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=12,
                classes=num_classes,
            )
        else:
            raise Exception("Regression not supported for ResNet18")
    elif model_architecture == "deeplabv3-efficientnet-b0":
        if class_designation is not None:
            model = smp.DeepLabV3(
                encoder_name="efficientnet-b0",
                encoder_weights=None,
                in_channels=12,
                classes=num_classes,
            )
        else:
            raise Exception("Regression not supported for EfficientNet-b0")
    elif model_architecture.startswith("deeplabv3-mobilenet_v2"):
        if class_designation is not None:
            model = smp.DeepLabV3(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=12,
                classes=num_classes,
            )
        else:
            raise Exception("Regression not supported for MobileNet-v2")
    elif model_architecture.startswith("deeplabv3-timm-mobilenetv3_small_minimal_100"):
        if class_designation is not None:
            model = smp.DeepLabV3(
                encoder_name="timm-mobilenetv3_small_minimal_100",
                encoder_weights=None,
                in_channels=12,
                classes=num_classes,
            )
        else:
            raise Exception(
                "Regression not supported for timm-mobilenetv3_small_minimal_100"
            )
    elif model_architecture.startswith("fcn_resnet50"):
        if class_designation is not None:
            model = fcn_resnet50(num_classes=num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(
                12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        else:
            raise Exception("Regression not supported for fcn-resnet50")
    elif model_architecture.startswith("deeplabv3_resnet50"):
        if class_designation is not None:
            model = deeplabv3_resnet50(num_classes=num_classes)
            model.backbone.conv1 = torch.nn.Conv2d(
                12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        else:
            raise Exception("Regression not supported for fcn-resnet50")
    elif model_architecture.startswith("deeplabv3_mobilenet_v3_large"):
        if class_designation is not None:
            model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)
            model.backbone["0"][0] = torch.nn.Conv2d(
                12, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
        else:
            raise Exception("Regression not supported for deeplabv3_mobilenet_v3_large")
    elif model_architecture.startswith("lraspp_mobilenet_v3_large"):
        if class_designation is not None:
            model = lraspp_mobilenet_v3_large(num_classes=num_classes)
            model.backbone["0"][0] = torch.nn.Conv2d(
                12, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
        else:
            raise Exception("Regression not supported for lraspp_mobilenet_v3_large")
    elif model_architecture.startswith("two_head_deeplabv3_resnet50"):
        if class_designation is not None:
            model = create_twohead_model(num_classes)
            model.backbone["0"][0] = torch.nn.Conv2d(
                12, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            )
        else:
            raise Exception("Regression not supported for lraspp_mobilenet_v3_large")
    else:
        raise Exception(f"Unknown model architecture {model_architecture}")

    if "no_replace_batchnorm" not in model_architecture:
        convert_batchnorm2d(model)
    if "use_groupnorm" in model_architecture:
        use_groupnorm(model)
    if "use_dropout" in model_architecture:
        model.segmentation_head = SegmentationHeadDropout(
            in_channels=model.decoder.out_channels,
            out_channels=num_classes,
            activation=None,
            kernel_size=1,
            upsampling=8,
        )

    model = model.to(device)
    if model_file is not None:
        # An entire path was passed in
        if model_file.startswith("/"):
            model.load_state_dict(torch.load(model_file, map_location=device))
        else:
            model_file_path = f"{model_folder}/{model_file}"
            model.load_state_dict(torch.load(model_file_path, map_location=device))
    return model


def mse_loss_with_nans(pred, label):
    # Custom loss function that changes removes any pixels that are 254 or 255 from consideration

    mask = label == -1
    # Multiply the flattened prediction by 253 to put it in the same range as the target
    pred = pred * 253

    loss = torch.mean((pred[~mask] - label[~mask]) ** 2)
    return loss


def get_criterion(class_designation, class_weights):
    if class_designation is None:
        return mse_loss_with_nans
    else:
        lf = torch.nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=-1,
            # We remove weights for now, since we will do pixel specific weights
            # weight=torch.FloatTensor(class_weights).to(device),
        )
        return lf


def get_optimizer(model, learning_rate, weight_decay=0):
    # Create adam optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return optimizer
