import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_save_folder = "/home/conrad/results/models/experiment44/visualize/test"

with open(f"{file_save_folder}/band_zero_results.json") as f:
    zero_results = json.load(f)
with open(f"{file_save_folder}/band_random_results.json") as f:
    random_results = json.load(f)


def generate_figure(
    title, filename, f, acc, show_legend, linecolor=None, show=False, save=False
):
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

    fig = plt.figure()
    ax = fig.add_subplot()
    bands_label = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8a",
        "B9",
        "B11",
        "B12",
    ]
    edge_colors = [
        "black",
        "blue",
        "green",
        "red",
        "black",
        "black",
        "black",
        "black",
        "black",
        "black",
        "black",
        "black",
    ]
    for band in zero_results.keys():
        i = int(band)
        value = f(random_results, band)
        value2 = f(zero_results, band)
        # avg_acc_zero = np.mean(r_zero["MulticlassAccuracy"])
        # avg_acc_random = np.mean(r_random["MulticlassAccuracy"])
        # ax.bar(int(band) * 3, avg_acc_zero, color="b")
        ax.bar(
            bands_label[i],
            value,
            color="lightgrey",
            edgecolor=edge_colors[i],
            # align="edge",
            # width=0.4,
        )
        """
        ax.bar(
            bands_label[i],
            value2,
            color="grey",
            edgecolor=edge_colors[i],
            align="edge",
            width=-0.4,
        )
        """

    if linecolor is None:
        ax.plot([-1, 12], [acc, acc], "--", label=f"No Bands Removed", linewidth=2.0)
    else:
        ax.plot(
            [-1, 12],
            [acc, acc],
            "--",
            label=f"No Bands Removed",
            linewidth=2.0,
            color=linecolor,
        )

    ax.set_ylim(0.0, 1.1)
    ax.set_xlim(-1, 12)
    ax.set_yticks(
        [0, 0.2, 0.4, 0.6, 0.8, acc, 1.0]
        if acc > 0.85
        else [0, 0.2, 0.4, 0.6, 0.75, acc, 1.0]
    )

    if show_legend:
        plt.legend()
    ax.set(xlabel="Randomized Band", ylabel="Average Accuracy")
    plt.title(title)

    if save:
        plt.savefig(f"./figures/{filename}.pdf", dpi=1000, bbox_inches="tight")
    if show:
        plt.show()


generate_figure(
    "Accuracy After Bands Randomized",
    "band_randomize_all_acc.pdf",
    lambda results, band: np.mean(results[band]["MulticlassAccuracy"]),
    0.8627,
    True,
    show=False,
    save=True,
)

generate_figure(
    "Accuracy After Bands Randomized for Low Risk HAB",
    "band_randomize_low_acc.pdf",
    lambda results, band: results[band]["MulticlassAccuracy"][0][0],
    0.8829,
    True,
    linecolor=np.array([149, 149, 149, 255]) / 255,
    show=False,
    save=True,
)


generate_figure(
    "Accuracy After Bands Randomized for Moderate Risk HAB",
    "band_randomize_moderate_acc.pdf",
    lambda results, band: results[band]["MulticlassAccuracy"][0][1],
    0.8276,
    True,
    linecolor=np.array([236, 173, 10, 255]) / 255,
    show=False,
    save=True,
)
generate_figure(
    "Accuracy After Bands Randomized for High Risk HAB",
    "band_randomize_high_acc.pdf",
    lambda results, band: results[band]["MulticlassAccuracy"][0][2],
    0.8776,
    True,
    linecolor=np.array([169, 0, 0, 255]) / 255,
    show=False,
    save=True,
)
