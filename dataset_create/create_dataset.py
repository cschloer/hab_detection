from download_and_process import download_and_process
from helpers import (
    get_cyan_url,
    TEMP_FOLDER,
    SAVE_FOLDER,
    ZIP_FILE_TRAIN,
    ZIP_FILE_TEST,
    get_api,
)
import zipfile
import datetime
import os
from archive import get_products
import json
import time
import threading
from threading import Thread
from sentinelsat import LTAError, LTATriggered, SentinelAPI


dl_ready = []
lock_dl_ready = threading.Lock()

trigger_list = []
lock_trigger_list = threading.Lock()

existing_prefixes = set()
lock_existing_prefixes = threading.Lock()


complete = []


def generate_file_prefix(id_, date):
    return f"{id_}_{str(date.year).zfill(4)}_{str(date.month).zfill(2)}_{str(date.day).zfill(2)}"


def manage_triggers(api, name):
    log_prefix = f"-- LTA Trigger Thread {name}: "
    waiting = {}

    def handle_online(r):
        is_online = api.is_online(r["uuid"])
        if is_online:
            with lock_dl_ready:
                print(f"{log_prefix}Adding {r['uuid']} to download queue")
                print(r)
                dl_ready.append(r)
                return True
        return False

    while True:
        # Check if previously triggered items are now online
        delete_waiting = []
        for k in waiting.keys():
            if handle_online(waiting[k]):
                delete_waiting.append(k)
        for k in delete_waiting:
            del waiting[k]

        ##########################################

        # Find new things to trigger
        while True:
            with lock_trigger_list:
                if len(trigger_list):
                    r = trigger_list.pop()
                else:
                    break

            if not handle_online(r):
                try:
                    # Trigger it!
                    print(f"{log_prefix}Triggering LTA for {r['uuid']}")
                    api.download(r["uuid"])
                except LTATriggered as e:
                    # Succesfully triggered
                    assert r["uuid"] not in waiting
                    waiting[r["uuid"]] = r

                except LTAError as e:
                    # No more trigger credits, add back to trigger list
                    with lock_trigger_list:
                        trigger_list.append(r)
                    # Break out of loop to sleep
                    # print(f"{log_prefix}Ran out of LTA trigger credits. Sleeping now")
                    break
        time.sleep(60)


def manage_downloads(api):
    log_prefix = f"-- Download Thread: "
    counter = 0
    while True:
        while True:
            r = None
            with lock_dl_ready:
                if len(dl_ready):
                    r = dl_ready.pop()
            if r is not None:
                is_online = api.is_online(r["uuid"])
                # Put it back in the queue
                if not is_online:
                    print(
                        f"{log_prefix}Putting a file back into the LTA trigger queue: {r['uuid']}"
                    )
                    with lock_trigger_list:
                        trigger_list.append(r)
                else:
                    with lock_existing_prefixes:
                        if r["file_prefix"] in existing_prefixes:
                            counter += 1
                            print(
                                f"{log_prefix}File with ID {r['id']} already exists in zip - {counter}"
                            )
                            continue
                        else:
                            existing_prefixes.add(r["file_prefix"])

                    try:
                        print(
                            f"{log_prefix}Downloading file {r['uuid']} - {counter + 1}"
                        )
                        download_and_process(
                            api,
                            r["uuid"],
                            r["id"],
                            r["window"],
                            r["cyan_id"],
                            r["date"],
                            ZIP_FILE_TRAIN
                            if r["designation"] == "train"
                            else ZIP_FILE_TEST,
                            f"{SAVE_FOLDER}/images/scenes/{r['id']}/{r['date'].year}-{r['date'].month}-{r['date'].day}",
                            log_prefix,
                        )
                        counter += 1

                    except Exception as e:
                        print(f"{log_prefix}GOT AN ERROR: {e}")
                        with lock_trigger_list:
                            with lock_existing_prefixes:
                                existing_prefixes.remove(r["file_prefix"])

                            trigger_list.append(r)
            else:
                # Sleep a little
                break
        time.sleep(30)


with open(f"{SAVE_FOLDER}/data.json", "r") as f:
    scenes = json.load(f)

api = get_api(
    os.environ.get("ESA_USER1").strip('"'),
    os.environ.get("ESA_PASSWORD1").strip('"'),
)

with lock_trigger_list:
    for scene in scenes:
        products = get_products(
            api,
            scene["window"],
            datetime.datetime(2019, 1, 1),
            datetime.datetime(2020, 12, 31),
        )
        for _, product in products.iterrows():
            date = product["beginposition"].to_pydatetime()
            uuid = product["uuid"]
            trigger_list.append(
                {
                    "date": date,
                    "file_prefix": generate_file_prefix(scene["id"], date),
                    "uuid": uuid,
                    "id": scene["id"],
                    "cyan_id": scene["cyan_id"],
                    "window": scene["window"],
                    "designation": scene["designation"],
                }
            )
    print(f"Found {len(trigger_list)} total items.")

with lock_existing_prefixes:
    with zipfile.ZipFile(ZIP_FILE_TRAIN, mode="a", compression=zipfile.ZIP_STORED) as z:
        # Files are in the format:
        # 1_1_X0001_Y0001_S050_2000_01_01*
        # This gives us the identifier
        existing_prefixes = set([n[:31] for n in z.namelist()])
    with zipfile.ZipFile(ZIP_FILE_TEST, mode="a", compression=zipfile.ZIP_STORED) as z:
        existing_prefixes.update(set([n[:31] for n in z.namelist()]))
print(existing_prefixes)

# LTA Thread 1
try:
    thread_triggers1 = Thread(
        target=manage_triggers,
        args=(api, "1"),
    )
    thread_triggers1.start()
except:
    print("Failed to make thread 1")

# LTA Thread 2
"""
try:
    api2 = get_api(
        os.environ.get("ESA_USER2").strip('"'),
        os.environ.get("ESA_PASSWORD2").strip('"'),
    )
    thread_triggers2 = Thread(
        target=manage_triggers,
        args=(api2, "2"),
    )
    thread_triggers2.start()
except:
    print("Failed to make thread 1")
"""

# Download thread
thread_downloads = Thread(
    target=manage_downloads,
    args=(api,),
)
thread_downloads.start()
"""
uuid = "36cde9b7-0c8d-42f5-96bd-0ab7c4cc4d10"
date = datetime.datetime(2019, 1, 6, 15, 56, 31, 24000)
id_ = "8_3_X0000_Y0800_S050"
cyan_id = "8_3"
window = [-76.97212808351101, 38.4252545714399, -76.83693062103936, 38.267412116899244]
download_and_process(
    api,
    uuid,
    id_,
    window,
    cyan_id,
    date,
    ZIP_FILE_TRAIN,
    f"{SAVE_FOLDER}/images/scenes/{id_}",
    "-- Download Thread: ",
)
"""

while True:
    time.sleep(100)
