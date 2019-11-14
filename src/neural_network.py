import numpy as np

from src.mathematics import sigmoid, log_loss
from src.util import add_bias, chunks


class NeuralNetwork:
    def __init__(self,
                 layers=[2, 4, 3, 2],
                 weights=None,
                 regularization_factor=0.25,
                 learning_rate=0.1,
                 batch_size=32,
                 epochs=1):
        self.layers = np.array(layers)
        self.num_layers = len(layers)
        self.weights = weights if weights is not None else self.generate_random_weights()
        self.regularization_factor = regularization_factor
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def generate_random_weights(self):
        return np.array([
            np.random.rand(input_num, layer_size + 1)
            for layer_size, input_num
            in zip(self.layers, self.layers[1:])
        ])

    def predict(self, inputs):
        for weights in self.weights:
            inputs = sigmoid(np.dot(weights, add_bias(inputs)))
        return inputs

    def _feedforward(self, instance):
        activations = []
        inputs = instance
        for weights in self.weights:
            bias_inputs = add_bias(inputs)
            activations.append(inputs)
            inputs = sigmoid(np.dot(weights, bias_inputs))
        activations.append(inputs)
        return activations

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

    def _layer_gradients(self, deltas, activations):
        return np.outer(deltas, add_bias(activations))

    def _gradients(self, expected, activations):
        deltas = self._deltas(expected, activations)
        return np.array([
            self._layer_gradients(layer_deltas, layer_activations)
            for layer_deltas, layer_activations
            in zip(deltas, activations)
        ])

    def _gradient_regularization(self):
        regularizations = []
        for weights in self.weights:
            regularization = np.zeros(weights.shape)
            regularization[:, 1:] = self.regularization_factor * weights[:, 1:]
            regularizations.append(regularization)
        return np.array(regularizations)

    def _regularized_mean_gradients(self, expected_batch, activations_batch):
        gradients_batch = np.array([
            self._gradients(expected, activations)
            for expected, activations
            in zip(expected_batch, activations_batch)
        ])

        n = len(expected_batch)
        regularized_mean_gradients = (gradients_batch.sum(0) + self._gradient_regularization()) / n
        return regularized_mean_gradients

    def _update_weights(self, gradients):
        self.weights = self.weights - self.learning_rate * gradients

    def _backpropagate(self, expected_batch, activations_batch):
        regularized_mean_gradients = self._regularized_mean_gradients(expected_batch, activations_batch)
        self._update_weights(regularized_mean_gradients)

    def train(self, examples, expected):
        total_examples = len(examples)
        total_expected = len(expected)
        if total_examples != total_expected:
            raise Exception('Examples and expected results must match in quantity!')

        examples_batches = chunks(examples, self.batch_size)
        expected_batches = chunks(expected, self.batch_size)

        for epoch in range(self.epochs):
            for examples_batch, expected_batch in zip(examples_batches, expected_batches):
                activations_batch = [self._feedforward(e) for e in examples_batch]
                self._backpropagate(expected_batch, activations_batch)

    def _loss_regularization(self, results):
        number_of_examples = len(results)

        regularization_acc = 0
        for layer_weights in self.weights:
            regularization_acc += np.power(layer_weights[:, 1:], 2).sum()

        return (self.regularization_factor / number_of_examples / 2 * regularization_acc)

    def loss(self, results, expected, regularize=True):
        losses = log_loss(results, expected)
        regularization = self._loss_regularization(results) if regularize else 0
        return losses + regularization
