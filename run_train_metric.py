from torch.utils.data import DataLoader

from helpers import log
from model import load_model
from dataset import get_image_dataset
from metrics import get_model_performance
from constants import (
    device,
    ZIP_PATH_TRAIN,
    TRAINING_BATCH_SIZE,
    MODEL_SAVE_FOLDER,
    MODEL_LOAD_PATH,
)

log(f"Loading datasets...")

train_dataset = get_image_dataset(ZIP_PATH_TRAIN)
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
