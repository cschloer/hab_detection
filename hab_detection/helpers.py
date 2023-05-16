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

import numpy as np
import numpy.lib.format
import struct

# Faster numpy load
def load(file):
    if type(file) == str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))
