import matplotlib.gridspec as gridspec
from helpers import (
    get_cyan_url,
    TEMP_FOLDER,
    SAVE_FOLDER,
    cyan_colormap,
    dataset_mean,
    dataset_std,
)
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import segmentation_models_pytorch as smp
from osgeo import ogr, osr, gdal
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import numpy as np
import subprocess
import os
import datetime
import requests
import torchvision.transforms as transforms
from download_and_process import (
    warp_and_crop,
    generate_sen2_geotiff,
    normalize_sen2,
    get_land_filter,
    get_cloud_filter,
)
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform_input = transforms.Compose([transforms.Normalize(dataset_mean, dataset_std)])


# Copy the load model code, imports not working for now...
def load_model(
    model_architecture,
    # model_file is None if we shouldn't load from a previous model.
    # Otherwise path starting with "/" or filename in model_folder starting without /
    model_file,
    model_folder,
    class_designation,
):
    num_classes = 2
    if class_designation is not None:
        num_classes = len(class_designation)
    if model_architecture.startswith("deeplabv3-resnet18"):
        if class_designation is not None:
            model = smp.DeepLabV3(
                encoder_name="resnet18",
                encoder_weights=None,
                in_channels=12,
                classes=num_classes,
            )
        else:
            raise Exception("Regression not supported for ResNet18")

    if "no_replace_batchnorm" not in model_architecture:
        convert_batchnorm2d(model)
    model = model.to(device)
    if model_file is not None:
        # An entire path was passed in
        if model_file.startswith("/") or model_file.startswith("./"):
            model.load_state_dict(torch.load(model_file, map_location=device))
        else:
            model_file_path = f"{model_folder}/{model_file}"
            model.load_state_dict(torch.load(model_file_path, map_location=device))
    return model


CYAN_AUTH_TOKEN = os.environ.get("CYAN_AUTH_TOKEN").strip('"')
region_id = "7_2"
start_year = 2019
# start_month = 7
# start_day = 15
start_month = 8
start_day = 20
# start_month = 8
# start_day = 14
# The window for the small river
window = (-78.97112276386518, 42.07467197025461, -78.84683992695112, 41.99893968677112)
date = datetime.datetime(start_year, start_month, start_day)
cyan_download_url = get_cyan_url(date, region_id)
date_id = f"{region_id}_{date.strftime('%j')}"
cyan_download_path = f"{TEMP_FOLDER}/{date_id}.tif"
temp_cyan = f"{TEMP_FOLDER}/_testing_temp_cyan_{date_id}.tif"
cropped_cyan = f"{TEMP_FOLDER}/_testing_temp_cyan2_{date_id}.tif"
if not os.path.exists(cyan_download_path):
    print("Downloading")
    headers = {"Authorization": f"Bearer {CYAN_AUTH_TOKEN}"}
    response = requests.get(cyan_download_url, headers=headers)
    with open(cyan_download_path, "wb") as f:
        f.write(response.content)

    # Convert projection system of cyan image
    cmd = [
        "gdalwarp",
        "-t_srs",
        "EPSG:4326",
        "-srcnodata",
        '"254"',
        "-dstnodata",
        '"255"',
        cyan_download_path,
        temp_cyan,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = [
        "gdal_translate",
        "-f",
        "GTiff",
        "-projwin",
        str(window[0]),
        str(window[1]),
        str(window[2]),
        str(window[3]),
        "-projwin_srs",
        "EPSG:4326",
        temp_cyan,
        cropped_cyan,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    print("Already exists")

# Must copy s3 files into this folder using s3 cp
output_sen2_path = f"{TEMP_FOLDER}/sen2_{date_id}"
input_sen2_path = f"{TEMP_FOLDER}/{date_id}"
cropped_sen2 = f"{output_sen2_path}/cropped.tif"
interpolated_cyan = f"{TEMP_FOLDER}/interpolated_{date_id}.tif"

if not os.path.exists(cropped_sen2):
    outputFullVrt = f"{output_sen2_path}/16Bit-AllBands.vrt"
    outputFullTif = f"{output_sen2_path}/unified.tif"
    temp_sen2 = f"{output_sen2_path}/temp_sen2.tif"
    cmd = [
        "gdalbuildvrt",
        "-resolution",
        "user",
        "-tr",
        "20",
        "20",
        "-separate",
        outputFullVrt,
    ]
    outputFiles = []
    bands = {
        "band_01": input_sen2_path + "/" + "B01.jp2",
        "band_02": input_sen2_path + "/" + "B02.jp2",
        "band_03": input_sen2_path + "/" + "B03.jp2",
        "band_04": input_sen2_path + "/" + "B04.jp2",
        "band_05": input_sen2_path + "/" + "B05.jp2",
        "band_06": input_sen2_path + "/" + "B06.jp2",
        "band_07": input_sen2_path + "/" + "B07.jp2",
        "band_08": input_sen2_path + "/" + "B08.jp2",
        "band_08A": input_sen2_path + "/" + "B8A.jp2",
        "band_09": input_sen2_path + "/" + "B09.jp2",
        # No band 10, because it is removed in 2A. Only used in atmospheric correction
        "band_11": input_sen2_path + "/" + "B11.jp2",
        "band_12": input_sen2_path + "/" + "B12.jp2",
    }
    for band_key in sorted(bands.keys()):
        band_path = bands[band_key]
        cmd.append(band_path)
        outputFiles.append(band_path)
    my_file = Path(outputFullVrt)
    if not my_file.is_file():
        print("Unifying to VRT")
        subprocess.call(cmd, stderr=subprocess.STDOUT)

    cmd = ["gdal_translate", "-of", "GTiff", outputFullVrt, outputFullTif]

    my_file = Path(outputFullTif)
    if not my_file.is_file():
        # file exists
        print("Unifying to TIF")
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # crop
    gdal.Warp(temp_sen2, outputFullTif, dstSRS="EPSG:4326")
    src = gdal.Open(temp_sen2)
    _, xres, _, _, _, yres = src.GetGeoTransform()

    cmd = [
        "gdal_translate",
        "-f",
        "GTiff",
        "-projwin",
        str(window[0]),
        str(window[1]),
        str(window[2]),
        str(window[3]),
        "-projwin_srs",
        "EPSG:4326",
        temp_sen2,
        cropped_sen2,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Create interpolated cyan
    cmd = [
        "gdalwarp",
        "-s_srs",
        "EPSG:4326",
        "-tr",
        str(abs(xres)),
        str(abs(yres)),
        "-r",
        "bilinear",
        "-tap",
        "-srcnodata",
        "255",
        cropped_cyan,
        interpolated_cyan,
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

with rasterio.open(cropped_sen2, "r") as ds:
    sen2_np = ds.read()  # read all raster values

model = load_model(
    "deeplabv3-resnet18#no_replace_batchnorm",
    "epoch_50.pt",
    f"/home/conrad/results/models/experiment44",
    [100, 200, 254],
)
height = sen2_np.shape[1]
width = sen2_np.shape[2]
ycrop = height % 8
xcrop = width % 8
sen2_np = sen2_np[:, 0 : height - ycrop, 0 : width - xcrop]
with torch.no_grad():
    model.eval()
    transformed_batch = torch.unsqueeze(
        transform_input(
            torch.from_numpy(sen2_np.astype(np.float32) / 10000),
        ),
        0,
    ).to(device, dtype=torch.float)
    pred = model(transformed_batch)  # make prediction
    if isinstance(pred, dict):
        pred = pred["out"]
    pred = pred.cpu().detach()
    pred_np = np.squeeze(torch.argmax(pred, dim=1, keepdim=False).cpu().numpy())


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
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

ax = axs[0][0]
sen2_img = normalize_sen2(sen2_np[1, :, :], sen2_np[2, :, :], sen2_np[3, :, :])
ax.set_title(f"Sentinel 2 Image")
ax.imshow(sen2_img)
ax.axis("off")

with rasterio.open(cropped_cyan, "r") as ds:
    cyan_np = ds.read()  # read all raster values
# cyan_reshaped =
cyan_np = np.squeeze(cyan_np)
ax = axs[0][1]
ax.set_title(f"Actual HAB Index")
ax.imshow(cyan_colormap[cyan_np])
ax.axis("off")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cmap = matplotlib.colors.ListedColormap(cyan_colormap[:-2] / 255)
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array([])
mappable.set_clim(0, 1)
colorbar = ax.get_figure().colorbar(
    mappable, cax=cax, orientation="vertical"
)  # , ticks=[0, 1, 2, 3, 4])
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(["0", "253"])
colorbar.ax.tick_params(axis="both", which="both", length=0)

class_designation = [100, 200, 254]
custom_colormap = np.copy(cyan_colormap)
prev_val = 0
used = list(range(len(cyan_colormap)))
for i, c in enumerate(class_designation):
    cur_color = cyan_colormap[c - 1 if i != 0 else 0]
    for j in range(c - prev_val):
        custom_colormap[prev_val + j] = cur_color
    prev_val = c


with rasterio.open(interpolated_cyan, "r") as ds:
    cyan_np = ds.read()  # read all raster values
# cyan_reshaped =
cyan_np = np.squeeze(cyan_np)
ax = axs[1][0]
ax.set_title(f"Actual HAB Class")
ax.imshow(custom_colormap[cyan_np])
ax.axis("off")

pred_masked = np.array(class_designation)[pred_np] - 1
cloud_mask = get_cloud_filter(sen2_np)
land_mask = get_land_filter(sen2_np)
pred_masked = np.where(land_mask, 255, pred_masked)
pred_masked = np.where(cloud_mask, 255, pred_masked)
ax = axs[1][1]
ax.set_title(f"Prediction HAB Class")
ax.imshow(custom_colormap[pred_masked])
ax.axis("off")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cmap = matplotlib.colors.ListedColormap(
    [
        "#959595",
        "#ecad0a",
        "#a90000",
    ]
)
mappable = cm.ScalarMappable(cmap=cmap)
mappable.set_array([])
mappable.set_clim(-0.5, len(class_designation) + 0.5)
colorbar = ax.get_figure().colorbar(
    mappable, cax=cax, orientation="vertical"
)  # , ticks=[0, 1, 2, 3, 4])
colorbar.set_ticks(np.linspace(0, len(class_designation), len(class_designation)))
colorbar.set_ticklabels(["0-99", "100-199", "200-253"])

plt.savefig(f"./small_water_body_comparison.pdf", dpi=500, bbox_inches="tight")
plt.show()
