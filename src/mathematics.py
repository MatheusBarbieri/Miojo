import numpy as np
from numba import vectorize


@vectorize()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@vectorize()
def _log_loss(x, y):
    if x == 0:
        x = 1e-15
    elif x == 1:
        x = 1 - 1e-15
    return -y * np.log(x) - (1 - y) * np.log(1 - x)


def log_loss(result, expected):
    all_loss = np.array([_log_loss(r, e).sum() for r, e in zip(result, expected)])
    return all_loss.mean()
