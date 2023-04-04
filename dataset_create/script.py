import os
import datetime
from urllib.request import urlretrieve
from helpers import (
    get_cyan_url,
    TEMP_FOLDER,
)


def download_cyan_geotiff(download_path, date, region_id):
    download_url = get_cyan_url(date, region_id)
    print(download_url)
    urlretrieve(download_url, download_path)
    return download_path


log_prefix = "6_5_X0400_Y0350_S050"
temp_folder = f"{TEMP_FOLDER}/{log_prefix}"
os.makedirs(temp_folder, exist_ok=True)
date = datetime.datetime(2019, 7, 1)
cyan_geotiff_path = download_cyan_geotiff(temp_folder, date, "6_2")
