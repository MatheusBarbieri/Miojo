import os
import pandas as pd
from time import time

from src.cross_validation import KFolds
from src.file_system import load_dataset
from src.metrics import ConfusionMatrix
from src.neural_network import NeuralNetwork
from src.util import (
    attributes_and_target,
    expected_to_neural_network,
    generate_structure,
    normalize_dataset,
    results_to_labels
)


class ParametersEvaluationRunner:
    def __init__(self,
                 args,
                 hidden_layers_formats=[
                     [3],
                     [10],
                     [100],
                     [10, 10],
                     [100, 100],
                     [10, 10, 10],
                     [100, 100, 100]
                 ],
                 regularization_factors=[0.1, 0.25],
                 learning_rates=[0.5, 0.1, 0.01, 0.001],
                 batch_sizes=[16, 32],
                 epochs=[300],
                 turns=1):
        self._datasets_paths = args.datasets_paths
        self._outputs_path = args.outputs_path
        self._k_folds = args.k_folds
        self._hidden_layers_formats = hidden_layers_formats
        self._regularization_factors = regularization_factors
        self._learning_rates = learning_rates
        self._batch_sizes = batch_sizes
        self._epochs = epochs
        self._turns = turns
        self._total_executions = self.total_executions()

    def total_executions(self):
        return (
            len(self._hidden_layers_formats) *
            len(self._regularization_factors) *
            len(self._learning_rates) *
            len(self._batch_sizes) *
            len(self._epochs) *
            len(self._datasets_paths) *
            self._turns
        )

    def execute(self):
        datasets = [load_dataset(path) for path in self._datasets_paths]
        normalized_datasets = []
        normalized_datasets = [normalize_dataset(data) for data in datasets]
        execution_count = 0

        for turn in range(self._turns):
            for epoch in self._epochs:
                for batch_size in self._batch_sizes:
                    for learning_rate in self._learning_rates:
                        for regularization in self._regularization_factors:
                            for layer_structure in self._hidden_layers_formats:
                                for dataset, data_path in zip(normalized_datasets, self._datasets_paths):
                                    execution_count += 1
                                    self.run(
                                        dataset,
                                        data_path,
                                        layer_structure,
                                        regularization,
                                        learning_rate,
                                        batch_size,
                                        epoch,
                                        turn,
                                        execution_count
                                    )

    def run(self, dataset, data_path, layer_structure, regularization,
            learning_rate, batch_size, epoch, turn, execution_count):
        start = time()
        identifier = int(start)

        k_folds = KFolds(dataset, self._k_folds, sampling='stratified')
        splits = k_folds.split_generator()

        examples, expected, _ = attributes_and_target(dataset)
        expected = expected_to_neural_network(expected)

        all_results = []
        total_loss = 0
        total_epochs = 0
        for index, (train_data, test_data) in enumerate(splits):
            split_start = time()
            print(f'>>> Running for split {index + 1} of {self._k_folds}')
            train_attributes, train_expected, train_columns = attributes_and_target(train_data)
            test_attributes, test_expected, test_columns = attributes_and_target(test_data)

            train_expected = expected_to_neural_network(train_expected)

            layers = generate_structure(train_attributes, train_expected, layer_structure)

            neural_network = NeuralNetwork(
                layers=layers,
                regularization_factor=regularization,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epoch,
                verbosity=1
            )

            neural_network.train(train_attributes.values, train_expected.values)
            split_end = time()

            split_time = split_end - split_start
            epochs_used = neural_network._ran_epochs
            total_epochs += epochs_used
            test_results = neural_network.predict(test_attributes.values)
            results = results_to_labels(test_results, test_columns).join(test_expected)
            network_loss = neural_network.cost_from_examples(examples.values, expected.values)
            total_loss += network_loss

            split_confusion_matrix = ConfusionMatrix(results)

            split_stats = [
                split_confusion_matrix, data_path, identifier, layers,
                network_loss, regularization, learning_rate,
                batch_size, epoch, epochs_used, turn, split_time
            ]

            self.save_split(*split_stats)

            all_results.append(results)

        end = time()
        total_time = end - start
        average_train_time = total_time / self._k_folds
        mean_loss = total_loss / self._k_folds

        confusion_matrix = ConfusionMatrix(pd.concat(all_results))
        confusion_matrix.show()

        stats = [
            confusion_matrix, data_path, layers, mean_loss,
            regularization, learning_rate, batch_size, epoch,
            turn, total_time, average_train_time, execution_count]

        self.print_info(*stats)
        self.save(*stats)

    def print_info(self, confusion_matrix, data_path, layers, mean_loss,
                   regularization, learning_rate, batch_size, epoch,
                   turn, total_time, average_train_time, execution_count):
        dataset_name = os.path.basename(data_path).split('.')[0]
        print('\x1b[2J\x1b[H')
        print(f'Execution {execution_count} of {self._total_executions}. [{dataset_name}][{turn + 1} run]')
        print((
            f'Time spent: {total_time:0.2f} seconds. ({average_train_time:0.2f} '
            f'seconds per train-set)'))
        print((
            f'\tEpochs: {epoch}\n\tBatch Size: {batch_size}\n\tLearning Rate: {learning_rate}\n\t'
            f'Regularization: {regularization}\n\tLayers: {layers}\n\tSystem Loss: {mean_loss}'))
        confusion_matrix.show()

    def _create_dir(self, path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                print(f'Could not create {path} path to save model.')
                raise e

    def save(self, confusion_matrix, data_path,
             layers, mean_loss,
             regularization, learning_rate, batch_size, epoch,
             turn, total_time, average_train_time, execution_count):

        self._create_dir(self._outputs_path)

        file_path = os.path.join(self._outputs_path, 'results.csv')
        file_exists = os.path.exists(file_path)

        with open(file_path, 'a+') as f:
            dataset = os.path.basename(data_path).split('.')[0]
            accuracy = confusion_matrix.accuracy()
            recall = confusion_matrix.macro_recall()
            precision = confusion_matrix.macro_precision()
            f_measure = confusion_matrix.macro_f_measure(1)

            if not file_exists:
                file_header = (
                    '"dataset","accuracy","recall","precision","f_measure",'
                    '"layers","mean_loss","regularization","learning_rate","batch_size",'
                    '"epoch","turn","total_time","average_train_time","execution_count"\r\n'
                )
                f.write(file_header)

            entry = (
                f'"{dataset}",{accuracy},{recall},{precision},{f_measure},'
                f'"{layers}",{mean_loss},{regularization},{learning_rate},{batch_size},'
                f'{epoch},{turn},{total_time},{average_train_time},{execution_count}\r\n'
            )

            f.write(entry)

    def save_split(self, confusion_matrix, data_path, identifier,
                   layers, network_cost, regularization, learning_rate,
                   batch_size, total_epoch, epochs, turn, total_time):

        self._create_dir(self._outputs_path)

        file_path = os.path.join(self._outputs_path, 'results_per_split.csv')
        file_exists = os.path.exists(file_path)

        with open(file_path, 'a+') as f:
            dataset = os.path.basename(data_path).split('.')[0]
            accuracy = confusion_matrix.accuracy()
            recall = confusion_matrix.macro_recall()
            precision = confusion_matrix.macro_precision()
            f_measure = confusion_matrix.macro_f_measure(1)

            if not file_exists:
                file_header = (
                    '"dataset","accuracy","recall","precision","f_measure",'
                    '"layers","network_cost","regularization","learning_rate","batch_size",'
                    '"total_epoch","epochs","turn","total_time","exec_id"\r\n'
                )
                f.write(file_header)

            entry = (
                f'"{dataset}",{accuracy},{recall},{precision},{f_measure},'
                f'"{layers}",{network_cost},{regularization},{learning_rate},{batch_size},'
                f'{total_epoch},{epochs},{turn},{total_time},{identifier}\r\n'
            )

            f.write(entry)
