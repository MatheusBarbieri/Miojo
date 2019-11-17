import numpy as np
import pickle

from src.mathematics import sigmoid, log_loss
from src.util import add_bias, chunks


class NeuralNetwork:
    def __init__(self,
                 layers=[2, 4, 3, 2],
                 weights=None,
                 regularization_factor=0.25,
                 momentum_factor=0.9,
                 learning_rate=0.1,
                 batch_size=32,
                 epochs=1,
                 show_loss=False):
        self.layers = np.array(layers)
        self.num_layers = len(layers)
        self.weights = weights if weights is not None else self.generate_random_weights()
        self.regularization_factor = regularization_factor
        self.momentum_factor = momentum_factor
        self.velocity = 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.show_loss = show_loss

    def generate_random_weights(self):
        return np.array([
            np.random.normal(loc=0, scale=1, size=(input_num, layer_size + 1))
            for layer_size, input_num
            in zip(self.layers, self.layers[1:])
        ])

    def predict(self, inputs):
        all_results = []
        for entry in inputs:
            for weights in self.weights:
                entry = sigmoid(np.dot(weights, add_bias(entry)))
            all_results.append(entry)
        return np.array(all_results)

    def _feedforward(self, instances):
        all_activations = []
        for instance in instances:
            activations = []
            inputs = instance
            for weights in self.weights:
                bias_inputs = add_bias(inputs)
                activations.append(inputs)
                inputs = sigmoid(np.dot(weights, bias_inputs))
            activations.append(inputs)
            all_activations.append(np.array(activations))
        return all_activations

    def _layer_deltas(self, layer_weights, deltas, activations):
        t_weights = np.transpose(layer_weights)
        return (np.dot(t_weights, deltas)[1:] * activations * (1 - activations))

    def _deltas(self, expected, activations):
        output_deltas = activations[-1] - expected

        deltas = [output_deltas]

        outter_deltas = output_deltas
        for index in range(self.num_layers - 2, 0, -1):
            outter_deltas = self._layer_deltas(self.weights[index], outter_deltas, activations[index])
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

    def _update_weights(self, veolocity):
        self.weights = self.weights - self.learning_rate * veolocity

    def _velocity(self, gradients):
        return self.momentum_factor * self.velocity + (1 - self.momentum_factor) * gradients

    def _backpropagate(self, expected_batch, activations_batch):
        regularized_mean_gradients = self._regularized_mean_gradients(expected_batch, activations_batch)
        velocity = self._velocity(regularized_mean_gradients)
        self.velocity = velocity
        self._update_weights(velocity)

    def train(self, examples, expected):
        total_examples = len(examples)
        total_expected = len(expected)
        if total_examples != total_expected:
            raise Exception('Examples and expected results must match in quantity!')

        examples_batches = chunks(examples, self.batch_size)
        expected_batches = chunks(expected, self.batch_size)

        for epoch in range(self.epochs):
            loss_str = ''
            if self.show_loss:
                current_loss = self.loss(self.predict(examples), expected)
                loss_str = f' [Current Loss {current_loss:0.3f}]'
            print(f'Running Epoch {epoch + 1} of {self.epochs}{loss_str}', end='\r')
            for batch_num, (examples_batch, expected_batch) in enumerate(zip(examples_batches, expected_batches)):
                activations_batch = self._feedforward(examples_batch)
                self._backpropagate(expected_batch, activations_batch)
        print('\nFinished training!')

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

    def save(self, path):
        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
