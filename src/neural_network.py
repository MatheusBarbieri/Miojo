import os
import numpy as np
import pickle

from src.mathematics import sigmoid, log_loss
from src.util import add_bias, chunks


class NeuralNetwork:
    def __init__(self,
                 layers=[2, 4, 3, 2],
                 weights=None,
                 regularization_factor=0.25,
                 beta_1=0.9,
                 beta_2=0.999,
                 learning_rate=0.1,
                 batch_size=32,
                 epochs=1,
                 verbosity=0,
                 stop_by_cost=True,
                 cost_change_limit=0.001,
                 iter_without_change=10):
        self._layers = np.array(layers)
        self._num_layers = len(layers)
        self._weights = weights if weights is not None else self.generate_random_weights()
        self._regularization_factor = regularization_factor
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._verbosity = verbosity

        self._stop_by_cost = stop_by_cost
        self._cost_change_limit = cost_change_limit
        self._iter_without_change = iter_without_change
        self._current_cost = 1000
        self._stuck_count = 0

        self._current_velocity = 0
        self._current_sgema = 0
        self._atualization_count = 1

    def generate_random_weights(self):
        weights = np.empty((self._num_layers - 1,), dtype=object)
        for index, (layer_size, input_num) in enumerate(zip(self._layers, self._layers[1:])):
            weights[index] = np.random.normal(loc=0, scale=1, size=(input_num, layer_size + 1))
        return weights

    def predict(self, inputs):
        all_results = []
        for entry in inputs:
            for weights in self._weights:
                entry = sigmoid(np.dot(weights, add_bias(entry)))
            all_results.append(entry)
        return np.array(all_results)

    def _feedforward(self, instances):
        all_activations = []
        for instance in instances:
            activations = []
            inputs = instance
            for weights in self._weights:
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
        for index in range(self._num_layers - 2, 0, -1):
            outter_deltas = self._layer_deltas(self._weights[index], outter_deltas, activations[index])
            deltas.append(outter_deltas)

        return deltas[::-1]

    def _layer_gradients(self, deltas, activations):
        return np.outer(deltas, add_bias(activations))

    def _gradients(self, expected, activations):
        deltas = self._deltas(expected, activations)
        gradients = np.empty((self._num_layers - 1,), dtype=object)

        for index, (layer_deltas, layer_activations) in enumerate(zip(deltas, activations)):
            gradients[index] = self._layer_gradients(layer_deltas, layer_activations)

        return gradients

    def _gradient_regularization(self):
        regularizations = np.empty((self._num_layers - 1,), dtype=object)

        for index, weights in enumerate(self._weights):
            regularization = np.zeros(weights.shape)
            regularization[:, 1:] = self._regularization_factor * weights[:, 1:]
            regularizations[index] = regularization

        return regularizations

    def _regularized_mean_gradients(self, expected_batch, activations_batch):
        gradients_batch = np.array([
            self._gradients(expected, activations)
            for expected, activations
            in zip(expected_batch, activations_batch)
        ])

        n = len(expected_batch)
        regularized_mean_gradients = (gradients_batch.sum(0) + self._gradient_regularization()) / n
        return regularized_mean_gradients

    def _update_weights(self, velocity, sgema, epsilon=1e-15):
        self._weights = self._weights - self._learning_rate / np.power(sgema + epsilon, 0.5) * velocity

    def _velocity(self, gradients):
        return self._beta_1 * self._current_velocity + (1 - self._beta_1) * gradients

    def _sgema(self, gradients):
        return self._beta_2 * self._current_sgema + (1 - self._beta_2) * np.power(gradients, 2)

    def _backpropagate(self, expected_batch, activations_batch):
        regularized_mean_gradients = self._regularized_mean_gradients(expected_batch, activations_batch)
        velocity = self._velocity(regularized_mean_gradients)
        sgema = self._sgema(regularized_mean_gradients)

        unbiased_velocity = velocity / (1 - np.power(self._beta_1, self._atualization_count))
        unbiased_sgema = sgema / (1 - np.power(self._beta_2, self._atualization_count))

        self._atualization_count += 1
        self._current_velocity = velocity
        self._current_sgema = sgema

        self._update_weights(unbiased_velocity, unbiased_sgema)

    def _should_stop_by_cost(self, new_cost):
        if self._current_cost - new_cost > self._cost_change_limit:
            self._stuck_count = 0
            self._current_cost = new_cost
        else:
            self._stuck_count += 1
        return self._stuck_count == self._iter_without_change

    def train(self, examples, expected):
        total_examples = len(examples)
        total_expected = len(expected)
        if total_examples != total_expected:
            raise Exception('Examples and expected results must match in quantity!')

        examples_batches = chunks(examples, self._batch_size)
        expected_batches = chunks(expected, self._batch_size)

        for epoch in range(self._epochs):
            if self._stop_by_cost or self._verbosity > 0:
                current_cost = self.cost_from_examples(examples, expected)
                should_stop = self._should_stop_by_cost(current_cost)
                if should_stop:
                    print(
                        '\nCost function won\'t change for',
                        f'{self._iter_without_change} iterations [{current_cost:0.3f}].',
                        'Finished Training.'
                        )
                    break
            if self._verbosity > 0:
                cost_str = f' [Current cost {current_cost:0.5f}]'
                print(f'Running Epoch {epoch + 1} of {self._epochs}{cost_str}{" "*10}', end='\r')

            for batch_num, (examples_batch, expected_batch) in enumerate(zip(examples_batches, expected_batches)):
                activations_batch = self._feedforward(examples_batch)
                self._backpropagate(expected_batch, activations_batch)

    def _cost_regularization(self, results):
        number_of_examples = len(results)

        regularization_acc = 0
        for layer_weights in self._weights:
            regularization_acc += np.power(layer_weights[:, 1:], 2).sum()

        return (self._regularization_factor / number_of_examples / 2 * regularization_acc)

    def cost_from_examples(self, examples, expected, regularize=True):
        predictions = self.predict(examples)
        return self.cost(predictions, expected, regularize)

    def cost(self, results, expected, regularize=True):
        losses = log_loss(results, expected)
        regularization = self._cost_regularization(results) if regularize else 0
        return losses + regularization

    def save(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except Exception as e:
                print(f'Could not create {dir_path} path to save model.')
                raise e

        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception as e:
                print(f'Cold not open {path}')
                raise e
