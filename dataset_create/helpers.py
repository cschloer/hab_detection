import os

CYAN_APP_KEY = os.environ.get("CYAN_APP_KEY")
TEMP_FOLDER = "/tmp/hab"
os.makedirs(TEMP_FOLDER, exist_ok=True)


def get_cyan_url(date, region_id):
    day_of_year = date.strftime("%j")
    year = date.strftime("%Y")
    return f"https://oceandata.sci.gsfc.nasa.gov/ob/getfile/L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{region_id}.tif?appkey={CYAN_APP_KEY}"
