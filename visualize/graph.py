import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# from pixel_count import train_pixel_count as pixel_count
from dist import (
    all_pixel_count,
    test_pixel_count,
    train_pixel_count,
)

"""
subset = "test"

text = "Entire"
if subset == "train":
    text = "Train"
    pixel_count = train_pixel_count
if subset == "test":
    text = "Test"
    pixel_count = test_pixel_count
"""

data = {
    "x": np.concatenate((range(254), range(254), range(254))),
    "vals": np.concatenate(
        (
            all_pixel_count / np.sum(all_pixel_count),
            train_pixel_count / np.sum(train_pixel_count),
            test_pixel_count / np.sum(test_pixel_count),
        )
    ),
    "subset": ["all"] * len(train_pixel_count)
    + ["train"] * len(test_pixel_count)
    + ["test"] * len(test_pixel_count),
    "class": np.concatenate(
        [
            ["0-99 HAB, low risk"] * 100
            + ["100 - 199 HAB, moderate risk"] * 100
            + ["200-253 HAB, high risk"] * 54
        ]
        * 3
    ),
}
print(len(data["x"]))
print(len(data["vals"]))
print(len(data["subset"]))
print(len(data["class"]))

W = 5.8  # Figure width in inches, approximately A4-width - 2*1.25in margin
plt.rcParams.update(
    {
        "figure.figsize": (W, W / (4 / 3)),  # 4:3 aspect ratio
        "font.size": 12,  # Set font size to 11pt
        "axes.labelsize": 12,  # -> axis labels
        "legend.fontsize": 12,  # -> legends
        "text.usetex": True,
        "font.family": "serif",
        "font.family": "Palatino",
        "font.weight": "bold",
        "text.latex.preamble": (  # LaTeX preamble
            r"\usepackage{lmodern}"
            # ... more packages if needed
        ),
    }
)


fig, ax = plt.subplots()
color = ["green"] * 100 + ["orange"] * 100 + ["red"] * 54
low = np.array([149, 149, 149, 255]) / 255
moderate = np.array([236, 173, 10, 255]) / 255
high = np.array([169, 0, 0, 255]) / 255
p = sns.lineplot(
    data=data,
    x="x",
    y="vals",
    hue="class",
    style="subset",
    ax=ax,
    palette=[low, moderate, high],
)
p.legend(loc="lower left")
ax.set_ylabel("Fraction of Occurances")
ax.set_xlabel("HAB Value")
ax.set_title(f"Pixel Count Fraction by HAB Level")
ax.set_yscale("log")
plt.savefig("pixel_count.pdf", dpi=1000, bbox_inches="tight")
plt.savefig("pixel_count.svg", format="svg")
