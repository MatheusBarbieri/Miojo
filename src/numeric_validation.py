from copy import deepcopy

import numpy as np

from src.neural_network import NeuralNetwork


class baseValidator:
    def __init__(self, neural_network: NeuralNetwork, example_batch, expected_batch):
        self._neural_network = neural_network
        self._example_batch = example_batch
        self._expected_batch = expected_batch

    def _neural_network_gradients(self):
        activations_batch = self._neural_network._feedforward(self._example_batch)
        return self._neural_network._regularized_mean_gradients(self._expected_batch, activations_batch)


class GradientNumericValidator(baseValidator):
    def __init__(self, neural_network: NeuralNetwork, example_batch, expected_batch, epsilon=0.000001):
        super().__init__(neural_network, example_batch, expected_batch)
        self._epsilon = epsilon
        self.neural_network_gradients = self._neural_network_gradients()
        self.numeric_gradients = self._numeric_gradients()

    def _numeric_gradients(self):
        new_gradients = [np.zeros(w.shape) for w in self._neural_network.weights]

        for layer_index, layer_weights in enumerate(self._neural_network.weights):
            for neuron_index, neuron_weights in enumerate(layer_weights):
                for weight_index, weight in enumerate(neuron_weights):

                    new_neural_network_1 = deepcopy(self._neural_network)
                    new_neural_network_1.weights[layer_index][neuron_index][weight_index] += self._epsilon
                    results_plus_epsilon = [new_neural_network_1.predict(example) for example in self._example_batch]
                    loss_plus_epsilon = new_neural_network_1.loss(results_plus_epsilon, self._expected_batch)

                    new_neural_network_2 = deepcopy(self._neural_network)
                    new_neural_network_2.weights[layer_index][neuron_index][weight_index] -= self._epsilon
                    results_minus_epsilon = [new_neural_network_2.predict(example) for example in self._example_batch]
                    loss_minus_epsilon = new_neural_network_2.loss(results_minus_epsilon, self._expected_batch)

                    resulting_gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * self._epsilon)
                    new_gradients[layer_index][neuron_index][weight_index] = resulting_gradient

        return new_gradients

    def mean_absolute_error_per_gradient(self):
        return np.absolute(np.absolute(self.numeric_gradients) - np.absolute(self.neural_network_gradients))

    def mean_absolute_error_per_layer(self):
        return np.array([
            errors.mean()
            for errors
            in self.mean_absolute_error_per_gradient()
        ])

    def mean_absolute_error(self):
        return self.mean_absolute_error_per_layer().mean()


class BackpropagationValidator(baseValidator):
    def __init__(self, neural_network: NeuralNetwork, example_batch, expected_batch):
        super().__init__(neural_network, example_batch, expected_batch)

    def show_gradients(self):
        all_gradients = self._neural_network_gradients()
        for layer_index, layer_gradients in enumerate(all_gradients):
            for neuron_index, neuron_gradients in enumerate(layer_gradients):
                for gradient_index, gradient in enumerate(neuron_gradients):
                    print(f'{gradient:0.5f}', end='')
                    if gradient_index != len(neuron_gradients) - 1:
                        print(', ', end='')
                if neuron_index != len(layer_gradients) - 1:
                    print('; ', end='')
                else:
                    print()
