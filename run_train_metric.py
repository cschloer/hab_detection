from torch.utils.data import DataLoader

from hab_detection.helpers import log
from hab_detection.model import load_model
from hab_detection.dataset import get_image_dataset
from hab_detection.metrics import get_model_performance
from hab_detection.constants import (
    device,
    ZIP_PATH_TRAIN,
    ZIP_PATH_TEST,
    TRAINING_BATCH_SIZE,
    MODEL_SAVE_FOLDER,
    MODEL_LOAD_PATH,
)

log(f"Loading datasets...")

train_dataset = get_image_dataset(ZIP_PATH_TEST)
train_loader = DataLoader(
    train_dataset,
    batch_size=TRAINING_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

log(f"Loading model...")
model = load_model(MODEL_LOAD_PATH)

log(f"Getting statistics...")
get_model_performance(model, train_loader)
