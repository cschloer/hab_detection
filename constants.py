import torch

ZIP_PATH_TRAIN = "/shared/datasets/hab/data/dataset4_train.zip"
ZIP_PATH_TEST = "/shared/datasets/hab/data/dataset4_test.zip"
MODEL_LOAD_PATH = ""
MODEL_SAVE_FOLDER = "/shared/datasets/hab/models/experiment1"

TRAINING_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LEARNING_RATE = 1e-5

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Dataset mean and std calculated in google colab (when values divided by 10,000)
dataset_mean = [
    0.04235201,
    0.04784579,
    0.05547756,
    0.04363304,
    0.05092298,
    0.05012781,
    0.05277098,
    0.05194422,
    0.05095484,
    0.05174823,
    0.03550878,
    0.02762647,
]
dataset_std = [
    0.05505946,
    0.05726197,
    0.0569017,
    0.0569323,
    0.06115186,
    0.07730392,
    0.0840756,
    0.08879628,
    0.08847726,
    0.10554088,
    0.06592033,
    0.05038361,
]
