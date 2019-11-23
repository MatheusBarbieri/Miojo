import time
import os

from src.file_system import load_dataset
from src.neural_network import NeuralNetwork
from src.util import (
    attributes_and_target,
    chunks,
    expected_to_neural_network,
    generate_structure,
    normalize_dataset,
)


class CostEvaluationNeuralNetwork(NeuralNetwork):
    def __init__(self,
                 layers=[2, 4, 3, 2],
                 weights=None,
                 regularization_factor=0.25,
                 beta_1=0.9,
                 beta_2=0.999,
                 learning_rate=0.1,
                 batch_size=32,
                 epochs=1,
                 verbosity=False,
                 basepath='results/cost/',
                 filename='cost.csv'):
        super().__init__(layers, weights, regularization_factor, beta_1, beta_2,
                         learning_rate, batch_size, epochs, verbosity)

        self._basepath = basepath
        self._filename = filename
        self._creation_timestamp = time.time()

    def _save_cost(self, cost, batches_count, examples_count, epoch, current_batch_size):
        if not os.path.exists(self._basepath):
            try:
                os.makedirs(self._basepath)
            except Exception as e:
                print(f'Could not create {self._basepath} path to save model.')
                raise e

        file_path = os.path.join(self._basepath, self._filename)
        file_exists = os.path.exists(file_path)
        with open(file_path, 'a+') as f:
            if not file_exists:
                file_header = (
                    '"cost","batches_count","examples_count","current_batch_size","batch_size",'
                    '"epoch","total_epochs","learning_rate","regularization","beta_1","beta_2",'
                    '"execution_id"\r\n'
                )
                f.write(file_header)

            entry = (
                f'{cost},{batches_count},{examples_count},{current_batch_size},{self._batch_size},{epoch},'
                f'{self._epochs},{self._learning_rate},{self._regularization_factor},{self._beta_1},{self._beta_2},'
                f'{self._creation_timestamp}\r\n'
            )
            f.write(entry)

    def train(self, examples, expected):
        examples_batches = chunks(examples, self._batch_size)
        expected_batches = chunks(expected, self._batch_size)

        total_batches_count = 0
        total_examples_count = 0
        batch_size = 0

        for epoch in range(self._epochs):
            for batch_num, (examples_batch, expected_batch) in enumerate(zip(examples_batches, expected_batches)):
                current_cost = self.cost_from_examples(examples, expected)
                cost_str = f' [Current cost {current_cost:0.3f}]{" "*20}'
                print(f'Running Epoch {epoch + 1} of {self._epochs}{cost_str}', end='\r')

                activations_batch = self._feedforward(examples_batch)
                self._save_cost(current_cost, total_batches_count, total_examples_count, epoch + 1, batch_size)

                batch_size = len(examples_batch)
                total_batches_count += 1
                total_examples_count += batch_size
                self._backpropagate(expected_batch, activations_batch)
        print('\nFinished evaluation!')


def execute(args):
    data = load_dataset(args.dataset_path)
    normalized_data = normalize_dataset(data)

    train_attributes, train_expected, train_columns = attributes_and_target(normalized_data)
    train_expected = expected_to_neural_network(train_expected)

    layers = generate_structure(train_attributes, train_expected, args.structure)
    filename = os.path.basename(args.dataset_path)

    neural_network = CostEvaluationNeuralNetwork(
        layers=layers,
        regularization_factor=args.regularization,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbosity=args.verbose,
        basepath=args.outputs_path,
        filename=filename
    )

    neural_network.train(train_attributes.values, train_expected.values)
