import numpy as np


def add_bias(instance):
    return np.insert(instance, 0, 1, axis=0)


def chunks(array, chunks_size):
    total = len(array)
    has_last_chunk = int(total % chunks_size != 0)
    iterations = total // chunks_size + has_last_chunk

    chunks = []
    for i in range(iterations):
        chunks.append(array[i * chunks_size: (i + 1) * chunks_size])

    return np.array(chunks)
