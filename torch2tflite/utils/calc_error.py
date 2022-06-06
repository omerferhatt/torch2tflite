import logging

import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calc_error")


def calc_error(result_base, result_converted):
    mse = ((result_base - result_converted) ** 2).mean(axis=None)
    mae = np.abs(result_base - result_converted).mean(axis=None)
    logger.info(f"MSE (Mean-Square-Error): {mse}\tMAE (Mean-Absolute-Error): {mae}")
