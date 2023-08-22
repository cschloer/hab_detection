from collections import OrderedDict
import torch
from torch import Tensor
from typing import Dict
from torchvision.models.resnet import ResNet, resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3_TwoHead(torch.nn.Module):
    """

    Args:
        backbone (torch.nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (torch.nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        second_classifier (torch.nn.Module): second classifier used during training
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        classifier: torch.nn.Module,
        second_classifier: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.second_classifier = second_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(
            x, size=input_shape, mode="bilinear", align_corners=False
        )
        result["out"] = x

        x2 = features["out"]
        x2 = self.second_classifier(x2)
        x2 = torch.nn.functional.interpolate(
            x2, size=input_shape, mode="bilinear", align_corners=False
        )
        result["out2"] = x2

        return result


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
) -> DeepLabV3_TwoHead:
    return_layers = {"layer4": "out"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    classifier = DeepLabHead(2048, num_classes)
    second_classifier = DeepLabHead(2048, num_classes)
    second_classifier[4] = torch.nn.Sequential(
        torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)),
        torch.nn.Sigmoid(),
    )
    return DeepLabV3_TwoHead(backbone, classifier, second_classifier)


def create_twohead_model(num_classes):
    weights_backbone = ResNet50_Weights.verify(None)
    backbone = resnet50(
        weights=weights_backbone, replace_stride_with_dilation=[False, True, True]
    )
    model = _deeplabv3_resnet(backbone, num_classes)
    return model
