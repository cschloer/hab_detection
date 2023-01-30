import logging
from constants import MODEL_SAVE_FOLDER, LOG_NAME

FORMAT = "%(asctime)s: %(message)s"
logging.basicConfig(
    level=logging.INFO, filename=f"{MODEL_SAVE_FOLDER}/{LOG_NAME}", format=FORMAT
)


def log(s):
    logging.info(s)
