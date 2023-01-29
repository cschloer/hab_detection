import logging
from constants import MODEL_SAVE_FOLDER

FORMAT = "%(asctime)s: %(message)s"
logging.basicConfig(
    level=logging.INFO, filename=f"{MODEL_SAVE_FOLDER}/log.txt", format=FORMAT
)


def log(s):
    logging.info(s)
