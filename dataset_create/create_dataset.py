from helpers import get_cyan_url, TEMP_FOLDER, SAVE_FOLDER
import datetime
import os
from archive import trigger_lta, get_products
import json
import time
import threading
from threading import Thread
from sentinelsat import LTAError, LTATriggered, SentinelAPI


dl_ready = []
lock_dl_ready = threading.Lock()

trigger_list = []
lock_trigger_list = threading.Lock()


complete = []


def manage_triggers(api, name):
    log_prefix = f"-- LTA Trigger Thread {name}: "
    waiting = {}

    def handle_online(r):
        is_online = api.is_online(r["uuid"])
        if is_online:
            with lock_dl_ready:
                print(f"{log_prefix}Adding {r['uuid']} to download queue")
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
                    counter += 1
                    print(f"{log_prefix}Downloading file {r['uuid']} - {counter}")
                    # TODO process
            else:
                # Sleep a little
                break
        time.sleep(30)


with open(f"{SAVE_FOLDER}/data.json", "r") as f:
    scenes = json.load(f)

with lock_trigger_list:
    for scene in scenes:
        products = get_products(
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
                    "uuid": uuid,
                    "cyan_id": scene["cyan_id"],
                    "window": scene["window"],
                    "designation": scene["designation"],
                }
            )
        break
    print(f"Found {len(trigger_list)} total items.")


def get_api(user, password):
    return SentinelAPI(
        user,
        password,
        "https://apihub.copernicus.eu/apihub",
    )


api = get_api(
    os.environ.get("ESA_USER1"),
    os.environ.get("ESA_PASSWORD1"),
)
try:
    thread_triggers1 = Thread(
        target=manage_triggers,
        args=(api, "1"),
    )
    thread_triggers1.start()
except:
    print("Failed to make thread 1")

try:
    api2 = get_api(
        os.environ.get("ESA_USER2"),
        os.environ.get("ESA_PASSWORD2"),
    )
    thread_triggers2 = Thread(
        target=manage_triggers,
        args=(api2, "2"),
    )
    thread_triggers2.start()
except:
    print("Failed to make thread 1")

thread_downloads = Thread(
    target=manage_downloads,
    args=(api,),
)


thread_downloads.start()

while True:
    time.sleep(100)
