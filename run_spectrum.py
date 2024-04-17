import json
import sys
from hab_detection.metrics import get_model_performance
from hab_detection.dataset import get_data, ImageData
from hab_detection.constants import (
    STRUCTURED_FOLDER_PATH_TEST,
)
from torch.utils.data import DataLoader
import numpy as np

with open("experiments.json", "r") as f:
    experiments = json.load(f)


if len(sys.argv) != 2:
    print(f"run_spectrum.py requires exactly 1 argument, the experiment id.")
    exit()


experiment_name = sys.argv[1]
if experiment_name not in experiments:
    print(f"Experiment {experiment_name} not in experiments.json")

e = experiments[experiment_name]
class_designation = e["class_designation"]


features, labels, _ = get_data(
    STRUCTURED_FOLDER_PATH_TEST,
)

dataset = ImageData(
    features,
    labels,
    class_designation,
    zip_path=None,
    randomize=False,
    transform=True,
    in_memory=False,
)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

print(class_designation)
print(len(loader))
# spectrums = np.zeros((len(class_designation), 12), dtype=np.float128)
spectrums = []
for i in range(len(class_designation)):
    spectrum = np.empty((12), dtype=object)
    spectrum[...] = [np.array([]) for _ in range(spectrum.shape[0])]
    spectrums.append(spectrum)

for batch_idx, (inputs, labels, _, _) in enumerate(loader):
    inputs = inputs.numpy()
    labels = labels.numpy()

    inputs = np.moveaxis(inputs, 0, -1)
    inputs = inputs.reshape((12, inputs.shape[-1] * 64 * 64))
    labels = labels.flatten()

    # print("INPUTS SHAPES", inputs.shape)
    # print("LABELS SHAPES", labels.shape)
    for i in range(len(class_designation)):
        spectrum = spectrums[i]

        mask = np.where(labels == i)[0]

        pixels = np.take(inputs, mask, axis=1)
        for b in range(12):
            # spectrum[b].extend(pixels[b])
            spectrum[b] = np.concatenate((spectrum[b], pixels[b, :]), axis=0)

        # sums = np.sum(pixels, 1)
        # spectrums[i, :] += sums

    if (batch_idx + 1) % 100 == 0:
        print(f"Ran through {batch_idx + 1} batches...")
    if batch_idx >= 600:
        print("Breaking")
        # continue
        # print(spectrums)
        # print(counts)
        break
        exit()

for i in range(len(class_designation)):
    print(f"Class {i} Mean")
    print([np.mean(b) for b in spectrums[i]])
    print(f"Class {i} STD")
    print([np.std(b) for b in spectrums[i]])
    print("--------------")
