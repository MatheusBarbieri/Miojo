import numpy as np


def generate_random_weights(layers):
    return np.array([np.random.rand(input_num, layer_size + 1) for layer_size, input_num in zip(layers, layers[1:])])


def add_bias(instance):
    return np.insert(instance, 0, 1, axis=0)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss_function():
    pass


class NeuralNetwork:
    def __init__(self, layers=[5, 5, 5, 2], weights=None):
        self.layers = layers
        self.weights = weights if weights is not None else generate_random_weights(layers)

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
