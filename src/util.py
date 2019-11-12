import numpy as np


def add_bias(instance):
    return np.insert(instance, 0, 1, axis=0)


def chunks(array, chunks_size):
    iterations = len(array) // chunks_size + 1

    chunks = []
    for i in range(iterations):
        chunks.append(array[i * chunks_size: (i + 1) * chunks_size])

    return np.array(chunks)
