import os
import logging


def mkdir(folder):
    try:
        os.mkdir(folder)
        logging.info('Created checkpoint directory')
    except OSError:
        pass
