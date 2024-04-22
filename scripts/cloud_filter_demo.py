import rasterio
import os
import requests
import numpy as np
from rasterio import plot
from rasterio.plot import show
from skimage import exposure
import subprocess
from pathlib import Path
from osgeo import ogr, osr, gdal
from PIL import Image
import matplotlib.pyplot as plt


cyan_colormap = np.array(
    [
        [149, 149, 149, 255],
        [71, 24, 106, 255],
        [71, 25, 107, 255],
        [71, 27, 109, 255],
        [71, 29, 111, 255],
        [71, 31, 112, 255],
        [71, 32, 113, 255],
        [71, 33, 114, 255],
        [71, 34, 115, 255],
        [70, 36, 116, 255],
        [70, 38, 118, 255],
        [70, 39, 119, 255],
        [70, 41, 120, 255],
        [70, 42, 121, 255],
        [70, 43, 122, 255],
        [69, 44, 123, 255],
        [69, 47, 124, 255],
        [69, 48, 125, 255],
        [68, 49, 126, 255],
        [68, 51, 126, 255],
        [68, 52, 127, 255],
        [68, 53, 128, 255],
        [67, 56, 129, 255],
        [66, 57, 130, 255],
        [66, 58, 130, 255],
        [66, 59, 131, 255],
        [65, 60, 131, 255],
        [65, 61, 132, 255],
        [64, 64, 133, 255],
        [64, 65, 133, 255],
        [63, 66, 134, 255],
        [63, 67, 134, 255],
        [62, 68, 134, 255],
        [62, 70, 135, 255],
        [61, 72, 136, 255],
        [60, 73, 136, 255],
        [60, 74, 136, 255],
        [60, 75, 136, 255],
        [59, 76, 137, 255],
        [59, 77, 137, 255],
        [58, 80, 137, 255],
        [57, 81, 138, 255],
        [57, 82, 138, 255],
        [56, 83, 138, 255],
        [56, 84, 138, 255],
        [55, 85, 138, 255],
        [54, 87, 139, 255],
        [54, 88, 139, 255],
        [53, 89, 139, 255],
        [53, 90, 139, 255],
        [52, 91, 139, 255],
        [52, 92, 139, 255],
        [51, 94, 140, 255],
        [50, 95, 140, 255],
        [50, 96, 140, 255],
        [49, 97, 140, 255],
        [49, 98, 140, 255],
        [48, 99, 140, 255],
        [48, 101, 140, 255],
        [47, 102, 140, 255],
        [47, 103, 140, 255],
        [46, 104, 140, 255],
        [46, 105, 140, 255],
        [45, 107, 141, 255],
        [45, 108, 141, 255],
        [44, 109, 141, 255],
        [44, 110, 141, 255],
        [43, 111, 141, 255],
        [43, 112, 141, 255],
        [42, 114, 141, 255],
        [42, 115, 141, 255],
        [41, 116, 141, 255],
        [41, 117, 141, 255],
        [41, 118, 141, 255],
        [40, 119, 141, 255],
        [39, 121, 141, 255],
        [39, 121, 141, 255],
        [39, 122, 141, 255],
        [38, 123, 141, 255],
        [38, 124, 141, 255],
        [38, 125, 141, 255],
        [37, 127, 141, 255],
        [37, 128, 141, 255],
        [36, 129, 141, 255],
        [36, 130, 140, 255],
        [35, 131, 140, 255],
        [35, 132, 140, 255],
        [34, 134, 140, 255],
        [34, 135, 140, 255],
        [34, 136, 140, 255],
        [33, 136, 140, 255],
        [33, 137, 140, 255],
        [33, 138, 140, 255],
        [32, 140, 139, 255],
        [32, 141, 139, 255],
        [31, 142, 139, 255],
        [31, 143, 139, 255],
        [31, 144, 139, 255],
        [30, 145, 139, 255],
        [30, 147, 138, 255],
        [30, 148, 138, 255],
        [30, 149, 138, 255],
        [29, 150, 137, 255],
        [29, 151, 137, 255],
        [29, 152, 137, 255],
        [29, 153, 136, 255],
        [29, 154, 136, 255],
        [29, 155, 136, 255],
        [29, 156, 135, 255],
        [29, 157, 135, 255],
        [29, 158, 135, 255],
        [30, 160, 134, 255],
        [30, 161, 133, 255],
        [30, 162, 133, 255],
        [31, 163, 132, 255],
        [31, 164, 132, 255],
        [32, 165, 132, 255],
        [33, 166, 131, 255],
        [34, 167, 130, 255],
        [34, 168, 129, 255],
        [35, 169, 129, 255],
        [36, 170, 128, 255],
        [37, 171, 128, 255],
        [39, 173, 126, 255],
        [40, 174, 126, 255],
        [41, 175, 125, 255],
        [42, 176, 124, 255],
        [43, 176, 124, 255],
        [45, 177, 123, 255],
        [47, 179, 121, 255],
        [49, 180, 121, 255],
        [50, 181, 120, 255],
        [52, 182, 119, 255],
        [53, 183, 118, 255],
        [56, 184, 117, 255],
        [58, 185, 116, 255],
        [60, 186, 115, 255],
        [61, 187, 114, 255],
        [63, 188, 113, 255],
        [65, 189, 112, 255],
        [68, 190, 110, 255],
        [70, 191, 109, 255],
        [72, 192, 108, 255],
        [74, 193, 107, 255],
        [76, 193, 106, 255],
        [78, 194, 104, 255],
        [82, 196, 102, 255],
        [84, 197, 101, 255],
        [86, 197, 100, 255],
        [88, 198, 99, 255],
        [90, 199, 97, 255],
        [93, 200, 96, 255],
        [97, 201, 94, 255],
        [99, 202, 92, 255],
        [102, 203, 91, 255],
        [104, 203, 90, 255],
        [106, 204, 88, 255],
        [108, 205, 87, 255],
        [113, 206, 84, 255],
        [115, 207, 83, 255],
        [118, 207, 81, 255],
        [120, 208, 80, 255],
        [123, 209, 78, 255],
        [125, 209, 77, 255],
        [130, 210, 74, 255],
        [133, 211, 72, 255],
        [135, 212, 70, 255],
        [138, 212, 69, 255],
        [140, 213, 67, 255],
        [143, 213, 66, 255],
        [148, 214, 62, 255],
        [150, 215, 61, 255],
        [153, 215, 59, 255],
        [156, 216, 57, 255],
        [158, 216, 55, 255],
        [161, 217, 54, 255],
        [166, 218, 50, 255],
        [169, 218, 49, 255],
        [172, 219, 47, 255],
        [174, 219, 45, 255],
        [177, 220, 43, 255],
        [180, 220, 42, 255],
        [185, 221, 38, 255],
        [188, 221, 37, 255],
        [190, 222, 35, 255],
        [193, 222, 33, 255],
        [196, 222, 32, 255],
        [198, 223, 30, 255],
        [203, 219, 27, 255],
        [209, 216, 24, 255],
        [215, 214, 21, 255],
        [218, 210, 21, 255],
        [220, 206, 19, 255],
        [223, 201, 18, 255],
        [225, 196, 16, 255],
        [227, 191, 15, 255],
        [229, 187, 14, 255],
        [231, 182, 12, 255],
        [234, 177, 11, 255],
        [236, 173, 10, 255],
        [238, 168, 8, 255],
        [240, 163, 7, 255],
        [242, 158, 5, 255],
        [245, 154, 4, 255],
        [247, 149, 3, 255],
        [249, 144, 1, 255],
        [251, 139, 0, 255],
        [254, 135, 0, 255],
        [254, 130, 0, 255],
        [254, 124, 0, 255],
        [254, 119, 0, 255],
        [254, 114, 0, 255],
        [254, 109, 0, 255],
        [254, 103, 0, 255],
        [254, 98, 0, 255],
        [254, 93, 0, 255],
        [254, 88, 0, 255],
        [254, 82, 0, 255],
        [254, 77, 0, 255],
        [254, 72, 0, 255],
        [254, 67, 0, 255],
        [254, 61, 0, 255],
        [254, 56, 0, 255],
        [254, 51, 0, 255],
        [254, 46, 0, 255],
        [254, 40, 0, 255],
        [254, 35, 0, 255],
        [254, 30, 0, 255],
        [254, 25, 0, 255],
        [254, 19, 0, 255],
        [254, 14, 0, 255],
        [254, 9, 0, 255],
        [254, 4, 0, 255],
        [253, 0, 0, 255],
        [248, 0, 0, 255],
        [243, 0, 0, 255],
        [237, 0, 0, 255],
        [232, 0, 0, 255],
        [227, 0, 0, 255],
        [222, 0, 0, 255],
        [216, 0, 0, 255],
        [211, 0, 0, 255],
        [206, 0, 0, 255],
        [201, 0, 0, 255],
        [195, 0, 0, 255],
        [190, 0, 0, 255],
        [185, 0, 0, 255],
        [180, 0, 0, 255],
        [174, 0, 0, 255],
        [169, 0, 0, 255],
        [169, 0, 0, 255],
        [169, 0, 0, 255],
        [169, 0, 0, 255],
        [169, 0, 0, 255],
        [229, 216, 202, 255],
        [0, 0, 0, 255],
    ]
)


def download_cyan():
    CYAN_AUTH_TOKEN = os.environ.get("CYAN_AUTH_TOKEN").strip('"')

    year = 2019
    day_of_year = 240
    region_id = "7_2"

    cyan_download_url = f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif"
    headers = {"Authorization": f"Bearer {CYAN_AUTH_TOKEN}"}
    response = requests.get(cyan_download_url, headers=headers)
    with open("./tmp/cyan.tif", "wb") as f:
        f.write(response.content)


def generate_all_bands(outputPath="./tmp"):
    bands = {
        "band_02": "./tmp/B02.jp2",
        "band_03": "./tmp/B03.jp2",
        "band_04": "./tmp/B04.jp2",
        "band_08": "./tmp/B08.jp2",
        "band_8a": "./tmp/B8A.jp2",
        "band_09": "./tmp/B09.jp2",
        "band_11": "./tmp/B11.jp2",
    }

    outputVrt = f"{outputPath}/output.vrt"
    outputTif = f"{outputPath}/output.tif"

    cmd = [
        "gdalbuildvrt",
        "-resolution",
        "user",
        "-tr",
        "20",
        "20",
        "-separate",
        outputVrt,
    ]
    outputFiles = []
    for band_key in sorted(bands.keys()):
        band_path = bands[band_key]
        cmd.append(band_path)
        outputFiles.append(band_path)
    my_file = Path(outputVrt)
    if not my_file.is_file():
        subprocess.call(cmd)

    cmd = ["gdal_translate", "-of", "GTiff", outputVrt, outputTif]

    my_file = Path(outputTif)
    if not my_file.is_file():
        # file exists
        subprocess.call(cmd)
    return outputTif


def warp_and_crop(
    base="./tmp",
    sen2_image_path="./tmp/output.tif",
    cyan_image_path="./tmp/cyan.tif",
):
    temp_sen2 = f"{base}/temp_sen2.tif"
    temp2_sen2 = f"{base}/temp2_sen2.tif"
    temp_cyan = base + "/temp_cyan.tif"
    temp2_cyan = base + "/temp2_cyan.tif"
    temp3_cyan = base + "/temp3_cyan.tif"

    with contextlib.suppress(FileNotFoundError):
        os.remove(temp_cyan)
        os.remove(temp2_cyan)
        os.remove(temp3_cyan)
        os.remove(temp_sen2)
        os.remove(temp2_sen2)

    window = (
        -83.34001881167653,
        41.95767060541368,
        -83.23702198550465,
        41.88052089743174,
    )
    gdal.Warp(temp_sen2, sen2_image_path, dstSRS="EPSG:4326")
    src = gdal.Open(temp_sen2)
    _, xres, _, _, _, yres = src.GetGeoTransform()
    print("XRES, YRES", xres, yres)

    # Convert projection system of cyan image
    cmd = [
        "gdalwarp",
        "-t_srs",
        "EPSG:4326",
        "-srcnodata",
        '"254"',
        "-dstnodata",
        '"255"',
        cyan_image_path,
        temp_cyan,
    ]
    subprocess.check_call(cmd)

    # Change x and y resolution of cyan image
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
        temp_cyan,
        temp2_cyan,
    ]
    subprocess.check_call(cmd)

    # Translate the cyan image
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
        temp2_cyan,
        temp3_cyan,
    ]
    subprocess.check_call(cmd)

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
        temp2_sen2,
    ]
    subprocess.check_call(cmd)


def normalize_sen2(red, green, blue):
    def normalize(arr):
        """Function to normalize an input array to 0-1"""
        arr_min = arr.min()
        arr_max = arr.max()
        return (arr - arr_min) / (arr_max - arr_min)

    img = np.dstack((normalize(red), normalize(green), normalize(blue)))

    # Increase contrast
    pixvals = img
    minval = np.percentile(pixvals, 5)
    maxval = np.percentile(pixvals, 95)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval)) * 1
    Image.fromarray(pixvals.astype(np.uint8))
    return pixvals


def get_cloud_filter(img):
    # A cloud index developed by Zhai et. al, formulated into code for sen2
    # https://doi.org/10.1016/j.isprsjprs.2018.07.006

    # The indices here are based on the processing done in a previous step
    blue = img[0, :, :]
    green = img[1, :, :]
    red = img[2, :, :]
    NIR = img[3, :, :]
    SWIR_1 = img[5, :, :]
    SWIR_2 = img[6, :, :]

    CI_1 = np.absolute(((NIR + 2 * SWIR_1) / (blue + green + red)) - 1)
    CI_2 = (blue + green + red + NIR + SWIR_1 + SWIR_2) / 6

    # Parameter value, as described in paper
    T1 = 1

    t2 = 1 / 10
    mean_CI_2 = np.mean(CI_2)
    # T2 is based on the image values
    T2 = mean_CI_2 + t2 * (np.max(CI_2) - mean_CI_2)

    test = CI_2 < T2

    cloud_filter = (CI_1 < T1) & (CI_2 > T2)

    return cloud_filter


def get_land_filter(img):
    green = img[1, :, :]
    band_8a = img[4, :, :]
    band_11 = img[6, :, :]
    land_filter = (band_8a > green) & (band_11 > green)
    return land_filter


def apply_cloud_and_land_filter(cyan_np, cloud_filter, land_filter):
    # The cyan image after applying the sen2 cloud filter to it
    cyan_np_land_filtered = np.where(
        (cyan_np != 255) & (land_filter),
        255,
        cyan_np,
    )
    cyan_np_cloud_land_filtered = np.where(
        (cyan_np_land_filtered != 254) & (cloud_filter),
        255,
        cyan_np_land_filtered,
    )

    return cyan_np_cloud_land_filtered


def visualize(sen2="./tmp/temp2_sen2.tif", cyan="./tmp/temp3_cyan.tif"):
    cube = gdal.Open(sen2)

    band_blue = cube.GetRasterBand(1)
    band_green = cube.GetRasterBand(2)
    band_red = cube.GetRasterBand(3)
    print(band_blue)
    print(band_green)

    img1 = band_red.ReadAsArray(0, 0, cube.RasterXSize, cube.RasterYSize)
    img2 = band_green.ReadAsArray(0, 0, cube.RasterXSize, cube.RasterYSize)
    img3 = band_blue.ReadAsArray(0, 0, cube.RasterXSize, cube.RasterYSize)

    img = normalize_sen2(img1, img2, img3)

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
    fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    ax1 = axs[0][0]
    ax1.set_title("Sentinel 2 MSI 2A product")
    ax1.imshow(img)
    ax1.axis("off")

    ax2 = axs[0][1]
    img2 = Image.open(cyan)
    ax2.set_title("CYAN HAB Product")
    ax2.imshow(img2)
    ax2.axis("off")

    # Run cloud and land filter on entire image to visualize if it worked
    with rasterio.open(sen2, "r") as ds:
        sen2_np = ds.read()  # read all raster values
    cloud_filter = get_cloud_filter(sen2_np)
    land_filter = get_land_filter(sen2_np)

    with rasterio.open(cyan, "r") as ds:
        cyan_np = ds.read()  # read all raster values

    cyan_filtered = apply_cloud_and_land_filter(cyan_np, cloud_filter, land_filter)

    # Need to create a custom color palette here in order to display it
    cyan_filtered_reshaped = cyan_filtered.reshape(
        cyan_filtered.shape[1], cyan_filtered.shape[2]
    )
    cyan_image = cyan_colormap[cyan_filtered_reshaped]

    # Creating an overlay that has pixel value [0,0,0,255] which is non opaque black
    black = np.concatenate(
        (
            np.full(
                (
                    3,
                    cyan_filtered_reshaped.shape[0],
                    cyan_filtered_reshaped.shape[1],
                ),
                0,
            ),
            np.full(
                (
                    1,
                    cyan_filtered_reshaped.shape[0],
                    cyan_filtered_reshaped.shape[1],
                ),
                255,
            ),
        ),
        axis=0,
    )

    # Creating an overlay that has pixel value [0,0,0,0], which makes it entirely opaque
    opaque = np.full(
        (
            4,
            cyan_filtered_reshaped.shape[0],
            cyan_filtered_reshaped.shape[1],
        ),
        0,
    )

    black_white_filter = np.where(
        cyan_filtered_reshaped == 255,
        black,
        opaque,
    )
    black_white_filter_reshaped = np.moveaxis(black_white_filter, 0, -1)
    ax3 = axs[1][0]
    ax3.set_title("Sentinel 2 with filter")
    ax3.imshow(img)
    ax3.imshow(
        black_white_filter_reshaped,
    )
    ax3.axis("off")

    ax4 = axs[1][1]
    ax4.set_title("Cyan with sen2 clouds and land")
    ax4.imshow(cyan_image)
    ax4.axis("off")

    plt.savefig(f"./cloud_filter_demo.pdf", dpi=1000, bbox_inches="tight")


download_cyan()
generate_all_bands()
warp_and_crop()
visualize()
