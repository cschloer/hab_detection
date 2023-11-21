from hab_detection.model import load_model
from hab_detection.dataset import get_image_dataset
from hab_detection.constants import STRUCTURED_FOLDER_PATH_TEST, device
import torch
from torch.utils.data import DataLoader
from torchviz import make_dot
import numpy as np

batch = torch.from_numpy(np.load("batch.npy"))

class_designation = [100, 200, 254]
model = load_model(
    "deeplabv3-resnet18#no_replace_batchnorm",
    "/home/conrad/Dropbox/Masters/masters_thesis/hab_detection/bundle/model.pt",
    None,
    class_designation,
)

yhat = model(batch)

make_dot(yhat, params=dict(list(model.named_parameters()))).render(
    "torchviz", format="png"
)

"""
# Code to grab a batch from the server, since torchviz needs an install (so we'll run it locally, but grab data from the server)
test_dataset = get_image_dataset(
    STRUCTURED_FOLDER_PATH_TEST,
    class_designation,
    subset=None,
    in_memory=False,
    use_zip=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)
batch = next(iter(test_loader))
np.save("batch.npy", batch[0].numpy())
"""
