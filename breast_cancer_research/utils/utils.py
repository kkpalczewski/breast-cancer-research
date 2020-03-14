import os
import logging
from functools import wraps


def mkdir(folder):
    try:
        os.mkdir(folder)
        logging.info('Created checkpoint directory')
    except OSError:
        pass

#TODO: change so that there could be multiple trainings on the same model
def counter_global_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.counter += 1  # executed every time the wrapped function is called
        return func(*args, **kwargs)

    wrapper.counter = 0  # executed only once in decorator definition time
    return wrapper
