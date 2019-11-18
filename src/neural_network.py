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
                 show_loss=False):
        self.layers = np.array(layers)
        self.num_layers = len(layers)
        self.weights = weights if weights is not None else self.generate_random_weights()
        self.regularization_factor = regularization_factor
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.show_loss = show_loss

        self.velocity = 0
        self.sgema = 0
        self.atualization_count = 1

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

    def _update_weights(self, velocity, sgema, epsilon=1e-15):
        self.weights = self.weights - self.learning_rate / np.power(sgema + epsilon, 0.5) * velocity

    def _velocity(self, gradients):
        return self.beta_1 * self.velocity + (1 - self.beta_1) * gradients

    def _sgema(self, gradients):
        return self.beta_2 * self.sgema + (1 - self.beta_2) * np.power(gradients, 2)

    def _backpropagate(self, expected_batch, activations_batch):
        regularized_mean_gradients = self._regularized_mean_gradients(expected_batch, activations_batch)
        velocity = self._velocity(regularized_mean_gradients)
        sgema = self._sgema(regularized_mean_gradients)

        unbiased_velocity = velocity / (1 - np.power(self.beta_1, self.atualization_count))
        unbiased_sgema = sgema / (1 - np.power(self.beta_2, self.atualization_count))

        self.atualization_count += 1
        self.velocity = velocity
        self.sgema = sgema

        self._update_weights(unbiased_velocity, unbiased_sgema)

    def train(self, examples, expected):
        total_examples = len(examples)
        total_expected = len(expected)
        if total_examples != total_expected:
            raise Exception('Examples and expected results must match in quantity!')

        examples_batches = chunks(examples, self.batch_size)
        expected_batches = chunks(expected, self.batch_size)

        for epoch in range(self.epochs):
            cost_str = ''
            if self.show_loss:
                predictions = self.predict(examples)
                current_cost = self.cost(predictions, expected)
                cost_str = f' [Current cost {current_cost:0.3f}]'
            print(f'Running Epoch {epoch + 1} of {self.epochs}{cost_str}', end='\r')
            for batch_num, (examples_batch, expected_batch) in enumerate(zip(examples_batches, expected_batches)):
                activations_batch = self._feedforward(examples_batch)
                self._backpropagate(expected_batch, activations_batch)
        print('\nFinished training!')

    def _cost_regularization(self, results):
        number_of_examples = len(results)

        regularization_acc = 0
        for layer_weights in self.weights:
            regularization_acc += np.power(layer_weights[:, 1:], 2).sum()

        return (self.regularization_factor / number_of_examples / 2 * regularization_acc)

    def cost(self, results, expected, regularize=True):
        losses = log_loss(results, expected)
        regularization = self._cost_regularization(results) if regularize else 0
        return losses + regularization

    def save(self, path):
        with open(path, 'wb+') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
