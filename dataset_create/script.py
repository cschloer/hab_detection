from helpers import get_cyan_url, TEMP_FOLDER

import os
import sys
from pprint import pprint
from tqdm import tqdm
from PIL import Image, ImageColor, ImageDraw
import numpy as np
from osgeo import gdal
import pandas as pd
from urllib.request import urlretrieve
import datetime
import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer


CYAN_SIZE = 2000
SCENE_SIZE = 100
NUM_DAYS = 3

year = 2019
region_id = "6_2"
date = datetime.datetime(year, 7, 30)
scenes = {}
counter = 0
with tqdm(total=((CYAN_SIZE / SCENE_SIZE) ** 2) * NUM_DAYS, file=sys.stdout) as pbar:
    while date.strftime("%Y") == "2019":
        cyan_download_url = get_cyan_url(date, region_id)
        date_id = f"{region_id}_{date.strftime('%j')}"
        cyan_download_path = f"{TEMP_FOLDER}/{date_id}.tif"
        if not os.path.exists(cyan_download_path):
            urlretrieve(cyan_download_url, cyan_download_path)

        """
        # Draw the cyan image, visualizing the scenes
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        cyan_image = Image.open(cyan_download_path).convert("RGBA")
        draw = ImageDraw.Draw(cyan_image)
        for x in range(0, CYAN_SIZE, SCENE_SIZE):
            # for y in range(0, 2000, 50):
            for y in range(0, CYAN_SIZE, SCENE_SIZE):
                draw.rectangle(
                    ((x, y), (x + SCENE_SIZE, y + SCENE_SIZE)),
                    outline=ImageColor.getrgb("red"),
                )
        ax.set_title("CYAN HAB Product, Visualization of Scenes")
        ax.imshow(cyan_image)
        ax.axis("off")
        plt.show()
        """

        # raster_info = gdal.Info(cyan_download_path)
        # print(raster_info)
        with rasterio.open(cyan_download_path, "r") as ds:
            for x in range(0, CYAN_SIZE, SCENE_SIZE):
                for y in range(0, CYAN_SIZE, SCENE_SIZE):
                    pbar.update(1)

                    scene_id = f"{region_id}_X{x}_Y{y}"
                    if scene_id in scenes and scenes[scene_id]["ignore"]:
                        continue

                    # read all raster values
                    cyan_np = np.squeeze(ds.read())[
                        x : x + SCENE_SIZE, y : y + SCENE_SIZE
                    ]
                    if scene_id not in scenes:
                        percent_land = np.count_nonzero(cyan_np == 254) / cyan_np.size
                        if percent_land > 0.6:
                            # Too much land, skip it
                            scenes[scene_id] = {
                                "ignore": True,
                            }
                            continue

                        top_left = ds.xy(x, y)
                        bottom_right = ds.xy(x + SCENE_SIZE, y + SCENE_SIZE)

                        transformer = Transformer.from_crs(ds.crs, "EPSG:4326")
                        x1, y1 = transformer.transform(top_left[0], top_left[1])
                        x2, y2 = transformer.transform(bottom_right[0], bottom_right[1])
                        window = (
                            x1,
                            y1,
                            x2,
                            y2,
                        )
                        scenes[scene_id] = {
                            "ignore": False,
                            "window": window,
                            "hab": [],
                        }

                    percent_water_hab = np.count_nonzero(
                        (cyan_np < 254) & (cyan_np != 0)
                    ) / np.count_nonzero(cyan_np != 254)
                    scenes[scene_id]["hab"].append(percent_water_hab)

        date = date + datetime.timedelta(days=1)
        counter += 1
        if counter >= NUM_DAYS:
            break
for key in scenes.keys():
    scene = scenes[key]
    if not scene["ignore"]:
        print(f"{key}: {round(np.mean(scene['hab']) * 100, 2)}%")
