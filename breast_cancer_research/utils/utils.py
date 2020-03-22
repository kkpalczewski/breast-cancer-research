import os
import logging
from functools import wraps
from time import time
from datetime import datetime
import numpy as np


def mkdir(folder):
    try:
        os.mkdir(folder)
        logging.info('Created checkpoint directory')
    except OSError:
        pass


def get_converted_timestamp():
    timestamp = time()
    converted_timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
    return converted_timestamp


# TODO: change so that there could be multiple trainings on the same model
def counter_global_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.counter += 1  # executed every time the wrapped function is called
        return func(*args, **kwargs)

    wrapper.counter = 0  # executed only once in decorator definition time
    return wrapper


def elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = time()
        result = func(*args, **kwargs)
        te = time()

        elapsed = te - ts

        print(f'Training complete in {elapsed // 60}m {elapsed % 60}s')
        return result

    return wrapper


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dataset = pickle.load(fo, encoding='bytes')
    return dataset


def hwc_2_chw(img):
    return np.transpose(img, (2, 0, 1))
