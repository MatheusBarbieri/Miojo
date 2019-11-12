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
    def __init__(self,
                 layers=[2, 4, 3, 2],
                 weights=None,
                 regularization_factor=0.25,
                 learning_rate=0.1,
                 batch_size=32):
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = weights if weights is not None else generate_random_weights(np.array(layers))
        self.regularization_factor = regularization_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, examples, expected):
        total_examples = len(examples)
        total_expected = len(expected)
        if total_examples != total_expected:
            raise Exception('Examples and expected results must match in quantity!')

        examples_batches = chunks(examples, self.batch_size)
        expected_batches = chunks(expected, self.batch_size)

        for examples_batch, expected_batch in zip(examples_batches, expected_batches):
            activations_batch = [self.feedforward(e) for e in examples_batch]
            self.backpropagate(expected_batch, activations_batch)

    def loss_regularization(self, x, y):
        number_of_examples = len(x)

        regularization_acc = 0
        for layer_weights in self.weights:
            regularization_acc += np.power(layer_weights[:, 1:], 2).sum()

        return (self.regularization_factor / number_of_examples / 2 * regularization_acc)

    def loss(self, x, y):
        losses = loss(x, y)
        regularization = self.loss_regularization(x, y)
        return losses + regularization

    def predict(self, instance):
        inputs = instance
        for weights in self.weights:
            bias_inputs = add_bias(inputs)
            inputs = sigmoid(np.dot(weights, bias_inputs))
        return inputs

    def feedforward(self, instance):
        activations = []
        inputs = instance
        for weights in self.weights:
            bias_inputs = add_bias(inputs)
            activations.append(inputs)
            inputs = sigmoid(np.dot(weights, bias_inputs))
        activations.append(inputs)
        return activations

    def _update_weights(self, gradients):
        self.weights = self.weights - self.learning_rate * gradients

    def backpropagate(self, expected_batch, activations_batch):
        gradients_batch = np.array([
            self._gradients(expected, activations)
            for expected, activations
            in zip(expected_batch, activations_batch)
        ])

        n = len(expected_batch)

        regularized_mean_gradients = (gradients_batch.sum(0) + self._gradient_regularization()) / n
        self._update_weights(regularized_mean_gradients)

    def _delta(self, layer_weights, deltas, activations):
        t_weights = np.transpose(layer_weights)
        return (np.dot(t_weights, deltas)[1:] * activations * (1 - activations))

    def _deltas(self, expected, activations):
        output_deltas = activations[-1] - expected

        deltas = [output_deltas]

        outter_deltas = output_deltas
        for index in range(self.num_layers - 2, 0, -1):
            outter_deltas = self._delta(self.weights[index], outter_deltas, activations[index])
            deltas.append(outter_deltas)

        return deltas[::-1]

    def _gradient(self, deltas, activations):
        return np.outer(deltas, add_bias(activations))

    def _gradient_regularization(self):
        regularizations = []
        for weights in self.weights:
            regularization = np.zeros(weights.shape)
            regularization[:, 1:] = self.regularization_factor * weights[:, 1:]
            regularizations.append(regularization)
        return np.array(regularizations)

    def _calculate_gradients(self, deltas, activations):
        return np.array([
            self._gradient(layer_deltas, layer_activations)
            for layer_deltas, layer_activations
            in zip(deltas, activations)
        ])

    def _gradients(self, expected, activations):
        deltas = self._deltas(expected, activations)
        return self._calculate_gradients(deltas, activations)
