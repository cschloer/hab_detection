import sys
import logging
from .constants import MODEL_SAVE_FOLDER, LOG_NAME

FORMAT = "%(asctime)s: %(message)s"
try:
    logging.basicConfig(
        level=logging.INFO, filename=f"{MODEL_SAVE_FOLDER}/{LOG_NAME}", format=FORMAT
    )
except:
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format=FORMAT
    )


def log(s):
    logging.info(s)
