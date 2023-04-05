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

total_available = 0
total_downloaded = 0
lock_total_downloaded = threading.Lock()


complete = []


def generate_file_prefix(id_, date):
    return f"{id_}_{str(date.year).zfill(4)}_{str(date.month).zfill(2)}_{str(date.day).zfill(2)}"


def manage_triggers(api, name):
    global total_downloaded
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
            try:
                if handle_online(waiting[k]):
                    delete_waiting.append(k)
            except:
                pass
        for k in delete_waiting:
            del waiting[k]

        ##########################################

        # Find new things to trigger
        while True:
            with lock_trigger_list:
                if len(trigger_list):
                    r = trigger_list.pop()
                    with lock_existing_prefixes:
                        if r["file_prefix"] in existing_prefixes:
                            with lock_total_downloaded:
                                total_downloaded += 1
                                print(
                                    f"{log_prefix}File with prefix {r['file_prefix']} already exists in zip - {total_downloaded}"
                                )
                            continue
                else:
                    break

            if not handle_online(r):
                try:
                    # Trigger it!
                    print(f"{log_prefix}Triggering LTA for {r['uuid']}")
                    api.download(r["uuid"])
                except LTATriggered as e:
                    # Succesfully triggered
                    if r["file_prefix"] in waiting:
                        print(
                            f"{log_prefix}FOUND FILE PREFIX TWICE: {r} ------------- {waiting[r['file_prefix']]}"
                        )
                    else:
                        waiting[r["file_prefix"]] = r

                except LTAError as e:
                    # No more trigger credits, add back to trigger list
                    with lock_trigger_list:
                        trigger_list.append(r)
                    # Break out of loop to sleep
                    # print(f"{log_prefix}Ran out of LTA trigger credits. Sleeping now")
                    break
                except Exception as e:
                    print(f"{log_prefix}THERE WAS AN ERROR", e)
                    with lock_trigger_list:
                        trigger_list.append(r)
        time.sleep(60)
        print(f"{log_prefix}Waking up in LTA thread...")


def manage_downloads(api, name):
    global total_downloaded
    log_prefix = f"-- Download Thread {name}: "
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
                            with lock_total_downloaded:
                                total_downloaded += 1
                                print(
                                    f"{log_prefix}File with file_prefix {r['file_prefix']} already exists in zip - {total_downloaded}"
                                )
                            continue
                        else:
                            existing_prefixes.add(r["file_prefix"])

                    try:
                        with lock_total_downloaded:
                            print(
                                f"{log_prefix}Downloading file {r['uuid']} - {total_downloaded + 1}"
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
                        with lock_total_downloaded:
                            total_downloaded += 1

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
        print(f"{log_prefix}Waking up in download thread...")


with open(f"{SAVE_FOLDER}/data.json", "r") as f:
    scenes = json.load(f)

api = get_api(
    os.environ.get("ESA_USER1").strip('"'),
    os.environ.get("ESA_PASSWORD1").strip('"'),
)

"""
print("Creating list of products to download")

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
total_available = len(trigger_list)

with lock_existing_prefixes:
    with zipfile.ZipFile(ZIP_FILE_TRAIN, mode="a", compression=zipfile.ZIP_STORED) as z:
        # Files are in the format:
        # 1_1_X0001_Y0001_S050_2000_01_01*
        # This gives us the identifier
        existing_prefixes = set([n[:31] for n in z.namelist()])
    with zipfile.ZipFile(ZIP_FILE_TEST, mode="a", compression=zipfile.ZIP_STORED) as z:
        existing_prefixes.update(set([n[:31] for n in z.namelist()]))
print(f"# Existing Prefixes: {len(existing_prefixes)}")

# LTA Thread 1
try:
    thread_triggers1 = Thread(
        target=manage_triggers,
        args=(api, "1"),
    )
    thread_triggers1.start()
except:
    print("Failed to make thread 1")

time.sleep(10)
# Download threads
thread_downloads1 = Thread(
    target=manage_downloads,
    args=(api, "1"),
)
thread_downloads1.start()

thread_downloads2 = Thread(
    target=manage_downloads,
    args=(api, "2"),
)
thread_downloads2.start()

thread_downloads3 = Thread(
    target=manage_downloads,
    args=(api, "3"),
)
thread_downloads3.start()
"""
uuid = "b4dbfcda-6a23-4af6-b3cd-cab5a618c781"
date = datetime.datetime(2020, 12, 8, 16, 37, 1, 24000)
id_ = "6_5_X0450_Y0300_S050"
cyan_id = "6_5"
window = [-90.0625607542466, 30.337830755522525, -89.91660792035341, 30.194472304251768]
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
"""
