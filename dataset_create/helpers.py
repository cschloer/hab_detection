import os

# from sentinelsat import SentinelAPI
import numpy as np

CYAN_APP_KEY = os.environ.get("CYAN_APP_KEY").strip('"')
TEMP_FOLDER = "/tmp/hab"
SAVE_FOLDER = os.environ.get("SAVE_FOLDER").strip('"')
ZIP_FILE_TRAIN = f"{SAVE_FOLDER}/dataset_train.zip"
ZIP_FILE_TEST = f"{SAVE_FOLDER}/dataset_test.zip"
os.makedirs(TEMP_FOLDER, exist_ok=True)


def get_api(user, password):
    return SentinelAPI(
        user,
        password,
        "https://apihub.copernicus.eu/apihub",
        show_progressbars=False,
    )


def get_cyan_url(date, region_id):
    day_of_year = date.strftime("%j")
    year = date.strftime("%Y")
    return f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif"
    return f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif?appkey={CYAN_APP_KEY}"


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
dataset_mean = [
    0.0367425,
    0.04384967,
    0.05643992,
    0.04752899,
    0.05311135,
    0.04531556,
    0.04827843,
    0.04772811,
    0.04578368,
    0.04437868,
    0.03240987,
    0.02406007,
]
dataset_std = [
    0.04898498,
    0.05039225,
    0.05055515,
    0.05064571,
    0.05497834,
    0.0692344,
    0.07566278,
    0.08037745,
    0.08150004,
    0.10058163,
    0.06478662,
    0.04916096,
]
