import os
from helpers import (
    get_cyan_url,
    TEMP_FOLDER,
    cyan_colormap,
    get_api,
)
from download_and_process import download_sen2

api = get_api(
    os.environ.get("ESA_USER1").strip('"'),
    os.environ.get("ESA_PASSWORD1").strip('"'),
)

temp_folder = f"{TEMP_FOLDER}/test"
os.makedirs(temp_folder, exist_ok=True)
download_sen2(
    api,
    temp_folder + "/",
    "1cef1d83-4971-4082-9e86-96d844946a86",
    attempt_if_offline=False,
)
