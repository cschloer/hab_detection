import os
import numpy as np
import zipfile
import re
import matplotlib.pyplot as plt
import seaborn as sns

mean = [
    np.array(
        [
            -0.0014762228940297236,
            -0.012032453748398694,
            -0.01526423359893638,
            -0.04640575860305045,
            -0.028756556288567547,
            0.07984194537254749,
            0.08560193683817391,
            0.09220575869959326,
            0.1079827908223083,
            0.10029787849486767,
            0.11493758257362922,
            0.1087296158413019,
        ]
    ),
    np.array(
        [
            -0.004258279254081685,
            -0.029675306443136734,
            0.014765486911331812,
            -0.03453557126533197,
            0.20101847461431233,
            0.32290194743750317,
            0.33390938892961225,
            0.3062806123356862,
            0.32017014644640485,
            0.28534779542090805,
            0.23993125061257614,
            0.21486682174695226,
        ]
    ),
    np.array(
        [
            0.024569747560471505,
            0.05115839223650708,
            0.26669075662455105,
            0.06421291304459703,
            0.5086494172058524,
            0.5518288554878145,
            0.544054041499038,
            0.5060126205900868,
            0.4803021324778966,
            0.39214072849190484,
            0.2644697039511689,
            0.21448916349604463,
        ]
    ),
]
std = [
    np.array(
        [
            1.0171187682741025,
            1.0343023557372704,
            1.0346545413213433,
            1.0508824214465293,
            1.0501354272066497,
            1.1050991510164476,
            1.1217037990392595,
            1.1317089449363875,
            1.122712326948279,
            1.0172403301064772,
            1.074960333017315,
            1.0368932705313623,
        ]
    ),
    np.array(
        [
            1.1744718550828137,
            1.1624701518599465,
            1.1388528815030008,
            1.1505741604931803,
            1.171550468357241,
            1.3071846677398946,
            1.3451802646352202,
            1.3457615281142608,
            1.3503007240273388,
            1.3006843993515245,
            1.2646962406570084,
            1.2325189361085396,
        ]
    ),
    np.array(
        [
            1.5282732621553436,
            1.5102017497555842,
            1.5197065297644312,
            1.489075979152363,
            1.549255496090968,
            1.4870688704683364,
            1.4787392531457784,
            1.48275822762672,
            1.457085413136254,
            1.4542763165907788,
            1.4098661161271864,
            1.4728649380280032,
        ]
    ),
]

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

for i, b in enumerate(bands_label):
    r = 4
    print(
        f"{b} & {round(mean[0][i], r)} & {round(std[0][i], r)} & {round(mean[1][i], r)} & {round(std[1][i], r)} & {round(mean[2][i], r)} & {round(std[2][i], r)} \\\\ \\hline"
    )

exit()

bands = [443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1610, 2190]

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
fill_between = False
plt.plot(bands, mean[0], color=np.array([149, 149, 149, 255]) / 255, label="0-99 HAB")
plt.plot(bands, mean[1], color=np.array([236, 173, 10, 255]) / 255, label="100-199 HAB")
plt.plot(bands, mean[2], color=np.array([169, 0, 0, 255]) / 255, label="200-253 HAB")

if fill_between:
    plt.plot(
        bands,
        mean[0] + std[0],
        color=np.array([149, 149, 149, 255]) / 255,
        alpha=0.5,
    )
    plt.plot(
        bands,
        mean[0] - std[0],
        color=np.array([149, 149, 149, 255]) / 255,
        alpha=0.5,
    )
    """
    plt.fill_between(
        bands,
        mean[0] + std[0],
        mean[0] - std[0],
        color=np.array([149, 149, 149, 255]) / 255,
        alpha=0.1,
    )
    plt.fill_between(
        bands,
        mean[2] + std[2],
        mean[2] - std[2],
        color=np.array([169, 0, 0, 255]) / 255,
        alpha=0.1,
    )
    plt.fill_between(
        bands,
        mean[1] + std[1],
        mean[1] - std[1],
        color=np.array([236, 173, 10, 255]) / 255,
        alpha=0.1,
    )
    """


for i, x in enumerate(bands):
    color = "grey"
    if i == 1:
        color = "blue"
    if i == 2:
        color = "green"
    if i == 3:
        color = "red"
    plt.axvline(x=x, linewidth=2.0, color=color, alpha=0.3)
# plt.ylim(-0.5, 1.5)

plt.legend(loc="upper right")
plt.title("Mean Spectral Signature of Each Class")
plt.xlabel("Central Wavelength (nm)")
plt.ylabel("Reflectance Value After Preprocessing")
# plt.show()
plt.savefig(f"./figures/spectrum.pdf", dpi=1000, bbox_inches="tight")
