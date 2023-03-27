import os
from sentinelsat import SentinelAPI

CYAN_APP_KEY = os.environ.get("CYAN_APP_KEY")
TEMP_FOLDER = "/tmp/hab"
SAVE_FOLDER = "/home/conrad/Dropbox/Masters/masters_thesis/data"
os.makedirs(TEMP_FOLDER, exist_ok=True)

api = SentinelAPI(
    "cschloer", os.environ.get("ESA_PASSWORD"), "https://apihub.copernicus.eu/apihub"
)


def get_cyan_url(date, region_id):
    day_of_year = date.strftime("%j")
    year = date.strftime("%Y")
    return f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif?appkey={CYAN_APP_KEY}"
