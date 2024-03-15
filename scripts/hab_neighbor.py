import base64
from tqdm import tqdm
import requests
from osgeo import gdal
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter
import datetime
import sys
import os
import rasterio
import matplotlib.pyplot as plt

# A script to help determine the distribution of neighbor pixels in CYAN images

CYAN_AUTH_TOKEN = os.environ.get("CYAN_AUTH_TOKEN").strip('"')


def get_cyan_url(date, region_id):
    day_of_year = date.strftime("%j")
    year = date.strftime("%Y")
    return f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif"


def eight_neighbor_average_convolve2d(x):
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    neighbor_sum = convolve2d(x, kernel, mode="same", boundary="fill", fillvalue=0)

    num_neighbor = convolve2d(
        np.ones(x.shape), kernel, mode="same", boundary="fill", fillvalue=0
    )

    return neighbor_sum / num_neighbor


TEMP_FOLDER = "/home/conrad/Dropbox/Masters/masters_thesis/hab_detection/scripts/temp"
region_ids = ["8_3", "6_2", "7_2", "7_5", "6_5"]
start_year = 2019
start_month = 6
start_day = 1
NUM_DAYS = 120
num_pixels_processed = 0
grouped_diffs = np.empty((254,), dtype=object)
for i in range(len(grouped_diffs)):
    grouped_diffs[i] = []
for h, region_id in enumerate(region_ids):
    date = datetime.datetime(start_year, start_month, start_day)
    print(f"Processing CYAN region {region_id}, {h+1}/{len(region_ids)}")
    with tqdm(total=NUM_DAYS, file=sys.stdout) as pbar:
        for j in range(NUM_DAYS):
            cyan_download_url = get_cyan_url(date, region_id)
            date_id = f"{region_id}_{date.strftime('%j')}"
            cyan_download_path = f"{TEMP_FOLDER}/{date_id}.tif"
            if not os.path.exists(cyan_download_path):
                headers = {"Authorization": f"Bearer {CYAN_AUTH_TOKEN}"}
                response = requests.get(cyan_download_url, headers=headers)
                with open(cyan_download_path, "wb") as f:
                    f.write(response.content)
            raster_info = gdal.Info(cyan_download_path)

            with rasterio.open(cyan_download_path, "r") as ds:
                # read all raster values
                cyan_np = np.squeeze(ds.read())

                cyan_np = np.where(cyan_np > 253, np.NaN, cyan_np)

                result = eight_neighbor_average_convolve2d(cyan_np)
                flat_cyan = cyan_np.flatten()
                filtered_cyan = flat_cyan[~np.isnan(flat_cyan)]
                flat_result = result.flatten()
                filtered_result = flat_result[~np.isnan(flat_cyan)]

                twice_filtered_cyan = filtered_cyan[~np.isnan(filtered_result)]
                twice_filtered_result = filtered_result[~np.isnan(filtered_result)]
                diff = np.abs(twice_filtered_result - twice_filtered_cyan)
                num_pixels_processed += len(diff)

                for i, cyan_val in enumerate(twice_filtered_cyan):
                    cyan_i = int(cyan_val)
                    grouped_diffs[cyan_i].append(diff[i])

                pbar.update(1)
            date = date + datetime.timedelta(days=1)
            if j > 5:
                continue
                # break
    continue
    # break

final_mean = np.zeros(len(grouped_diffs))
final_std = np.zeros(len(grouped_diffs))
for i in range(len(grouped_diffs)):
    final_mean[i] = np.mean(grouped_diffs[i])
    final_std[i] = np.std(grouped_diffs[i])
print(final_mean)
print(final_std)

print(f"Processed {num_pixels_processed} pixels")


lower_bound = np.where((final_mean - final_std) > 0, final_mean - final_std, 0)
upper_bound = final_mean + final_std

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

plt.plot(range(len(grouped_diffs)), final_mean)
plt.fill_between(range(len(grouped_diffs)), lower_bound, upper_bound, alpha=0.3)
# plt.tight_layout()
plt.title("Mean HAB Difference From Neighbors")
plt.xlabel("HAB index")
plt.xticks([0, 50, 100, 150, 200, 253], [0, 50, 100, 150, 200, 253])
plt.ylabel("Mean Difference")
plt.savefig(f"./hab_neighbor_mean_8.pdf", dpi=1000, bbox_inches="tight")
