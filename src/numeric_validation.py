from copy import deepcopy

import numpy as np

from src.neural_network import NeuralNetwork


def neural_network_gradients(neural_network: NeuralNetwork, example_batch, expected_batch):
    activations_batch = [neural_network.feedforward(e) for e in example_batch]
    return neural_network._regularized_mean_gradients(expected_batch, activations_batch)


def numeric_gradients(neural_network: NeuralNetwork, example_batch, expected_batch, epsilon):
    new_gradients = [np.zeros(w.shape) for w in neural_network.weights]

    for layer_index, layer_weights in enumerate(neural_network.weights):
        for neuron_index, neuron_weights in enumerate(layer_weights):
            for weight_index, weight in enumerate(neuron_weights):

                new_neural_network_1 = deepcopy(neural_network)
                new_neural_network_1.weights[layer_index][neuron_index][weight_index] += epsilon
                results_plus_epsilon = [new_neural_network_1.predict(example) for example in example_batch]
                loss_plus_epsilon = new_neural_network_1.loss(results_plus_epsilon, expected_batch)

                new_neural_network_2 = deepcopy(neural_network)
                new_neural_network_2.weights[layer_index][neuron_index][weight_index] -= epsilon
                results_minus_epsilon = [new_neural_network_2.predict(example) for example in example_batch]
                loss_minus_epsilon = new_neural_network_2.loss(results_minus_epsilon, expected_batch)

                resulting_gradient = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)
                new_gradients[layer_index][neuron_index][weight_index] = resulting_gradient

    return new_gradients


def gradient_numeric_validation(neural_network: NeuralNetwork, example_batch, expected_batch, epsilon=0.000001):
    nn_gradients = neural_network_gradients(neural_network, example_batch, expected_batch)
    nn_numeric_gradients = numeric_gradients(neural_network, example_batch, expected_batch, epsilon)
    return nn_gradients, nn_numeric_gradients
