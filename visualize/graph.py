import numpy as np
from matplotlib import pyplot as plt

# from dist import train_dist as dist
from dist import all_dist as dist, test_dist, train_dist

subset = ""

text = "Entire"
if subset == "train":
    text = "Train"
    dist = train_dist
if subset == "test":
    text = "Test"
    dist = test_dist

fig, ax = plt.subplots()
color = ["green"] * 100 + ["orange"] * 100 + ["red"] * 54
low = np.array([149, 149, 149, 255]) / 255
moderate = np.array([236, 173, 10, 255]) / 255
high = np.array([169, 0, 0, 255]) / 255
# moderate =
ax.plot(range(100), dist[:100], color=low, label="0-99 HAB, low risk")
ax.plot(
    range(100, 200), dist[100:200], color=moderate, label="100-199 HAB, moderate risk"
)
ax.plot(range(200, 254), dist[200:254], color=high, label="200-253 HAB, high risk")
ax.set_yscale("log")
ax.set_ylabel("Distribution Ratio")
ax.set_xlabel("HAB Value")
ax.set_title(f"Pixel Distribution by HAB Level in the {text} Dataset")
ax.set_ylim([0, 10e9])
ax.legend()
plt.show()
