import pandas as pd

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


def execute(args):
    data = load_dataset(args.dataset_path)
    normalized_data = normalize_dataset(data)

    k_folds = KFolds(normalized_data, args.k_folds, sampling='stratified')
    splits = k_folds.split_generator()

    all_results = []
    for train_data, test_data in splits:
        train_attributes, train_expected, train_columns = attributes_and_target(train_data)
        test_attributes, test_expected, test_columns = attributes_and_target(test_data)

        train_expected = expected_to_neural_network(train_expected)

        layers = generate_structure(train_attributes, train_expected, args.structure)

        neural_network = NeuralNetwork(
            layers=layers,
            regularization_factor=args.regularization,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            show_cost=args.verbose
        )

        neural_network.train(train_attributes.values, train_expected.values)
        test_results = neural_network.predict(test_attributes.values)
        results = results_to_labels(test_results, test_columns).join(test_expected)
        all_results.append(results)

    confusion_matrix = ConfusionMatrix(pd.concat(all_results))
    confusion_matrix.show()
