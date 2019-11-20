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


def normalize(examples):
    max_value = examples.max(axis=0)
    min_value = examples.min(axis=0)

    examples_min = examples - min_value
    max_min = max_value - min_value

    return np.array(
        np.divide(examples_min, max_min, out=np.zeros_like(examples_min), where=(max_min != 0))
    )
