from hab_detection.model import load_model
from hab_detection.dataset import get_image_dataset
from hab_detection.constants import STRUCTURED_FOLDER_PATH_TEST
from torch.utils.data import DataLoader

class_designation = [100, 200, 254]
model = load_model(
    "deeplabv3-resnet18#no_replace_batchnorm",
    "/home/conrad/Dropbox/Masters/masters_thesis/models/experiment44/epoch_50.pt",
    None,
    class_designation,
)

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
print("BATCH", batch)
yhat = model(batch.text)
print("YHAT", yhat)
