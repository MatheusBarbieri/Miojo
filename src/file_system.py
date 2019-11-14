import numpy as np


def load_network_from_text(path):
    with open(path) as f:
        regularization = float(f.readline())
        layers = np.array([int(num) for num in f.read().split()])
        return regularization, layers


def load_dataset_from_text(path):
    with open(path) as f:
        examples = []
        results = []
        for line in f.readlines():
            example, result = line.split(';')

            examples.append(np.array(
                [float(x) for x in example.split(',')])
            )
            results.append(np.array(
                [float(x) for x in result.split(',')])
            )

        return np.array(examples), np.array(results)


def load_weights_from_text(path):
    with open(path) as f:
        neural_network_weights = []
        for line in f.readlines():
            layer_weights_str = line.split(';')
            layer_weights = []
            for neuron_weights_str in layer_weights_str:
                weights = [float(weight) for weight in neuron_weights_str.split(',')]
                layer_weights.append(np.array(weights))
            neural_network_weights.append(np.array(layer_weights))
        return np.array(neural_network_weights)
