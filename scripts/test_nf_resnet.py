import timm
import torch
from nfnets import ScaledStdConv2d
import segmentation_models_pytorch as smp

"""
model = timm.create_model("hf-hub:timm/nf_resnet50.ra2_in1k", pretrained=False)
model.stem.conv = ScaledStdConv2d(
    12, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)
)
"""
model = smp.DeepLabV3(
    encoder_name="resnet18",
    # encoder_name="tu-nf_resnet26",
    encoder_weights=None,
    in_channels=12,
    classes=5,
)

print(model)
