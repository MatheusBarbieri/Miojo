import numpy as np
from numba import vectorize


@vectorize()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@vectorize()
def _log_loss(x, y, epsilon=1e-15):
    min_value = epsilon
    max_value = 1 - epsilon
    x = np.max(np.min(x, max_value), min_value)
    return -y * np.log(x) - (1 - y) * np.log(1 - x)


def log_loss(result, expected):
    all_loss = np.array([_log_loss(r, e).sum() for r, e in zip(result, expected)])
    return all_loss.mean()


normalize_default_min = 1e-15
normalize_default_max = 1 - 1e-15


def normalize(examples, min_value=normalize_default_min, max_value=normalize_default_max):
    return np.array(
        min_value + (examples - examples.min()) * (max_value - min_value) / (examples.max() - examples.min())
    )
