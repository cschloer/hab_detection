import sys
import logging
from .constants import MODEL_SAVE_BASE_FOLDER, LOG_NAME

FORMAT = "%(asctime)s: %(message)s"


def set_config(experiment_name):
    if experiment_name == "test":
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
            format=FORMAT,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            filename=f"{MODEL_SAVE_BASE_FOLDER}/{experiment_name}/{LOG_NAME}",
            format=FORMAT,
        )


def log(s):
    logging.info(s)
