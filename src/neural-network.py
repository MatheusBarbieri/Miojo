import numpy as np
from numba import vectorize


def generate_random_weights(layers):
    return np.array([np.random.rand(input_num, layer_size + 1) for layer_size, input_num in zip(layers, layers[1:])])


def add_bias(instance):
    return np.insert(instance, 0, 1, axis=0)


@vectorize()
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@vectorize()
def _loss(x, y):
    if x == 0:
        x = 1e-15
    elif x == 1:
        x = 1 - 1e-15
    return -y * np.log(x) - (1 - y) * np.log(1 - x)


def loss(result, expected):
    all_loss = np.array([_loss(r, e).sum() for r, e in zip(result, expected)])
    return all_loss.mean()


def chunks(array, chunks_size):
    iterations = len(array) // chunks_size + 1

    chunks = []
    for i in range(iterations):
        chunks.append(array[i * chunks_size: (i + 1) * chunks_size])

    return np.array(chunks)


class NeuralNetwork:
    def __init__(self, layers=[5, 5, 5, 2], weights=None):
        self.layers = layers
        self.weights = weights if weights is not None else generate_random_weights(np.array(layers))

    def train(self, batches_size=10):
        pass

    def predict(self, instance):
        inputs = instance
        for weights in self.weights:
            bias_inputs = add_bias(inputs)
            inputs = sigmoid(np.dot(weights, bias_inputs))
        return inputs

    def feedforward(self, instance):
        inputs = instance
        activations = []
        for weights in self.weights:
            bias_inputs = add_bias(inputs)
            inputs = sigmoid(np.dot(weights, bias_inputs))
            activations.append(inputs)
        return activations

    def backpropagation(self):
        pass
