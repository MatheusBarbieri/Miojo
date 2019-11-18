import numpy as np
from numba import vectorize

epsilon = 1e-15


@vectorize()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@vectorize()
def _log_loss(x, y):
    if x == 0:
        x = epsilon
    elif x == 1:
        x = 1 - epsilon
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))


def log_loss(result, expected):
    all_loss = np.array([_log_loss(r, e).sum() for r, e in zip(result, expected)])
    return all_loss.mean()


normalize_default_min = epsilon
normalize_default_max = 1 - epsilon


def normalize(examples, min_value=normalize_default_min, max_value=normalize_default_max):
    return np.array(
        min_value + (examples - examples.min(axis=0))
        * (max_value - min_value) / (examples.max(axis=0) - examples.min(axis=0))
    )
