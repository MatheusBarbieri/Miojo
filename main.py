import pandas as pd

from src.command_line import get_args
from src.cross_validation import KFolds
from src.file_system import load_model_from_text, load_dataset_from_text, load_dataset
from src.metrics import ConfusionMatrix
from src.neural_network import NeuralNetwork
from src.numeric_validation import BackpropagationValidator, GradientNumericValidator
from src.util import (
    attributes_and_target,
    expected_to_neural_network,
    generate_structure,
    get_attributes,
    normalize_dataset,
    results_to_labels
)


def main():
    args = get_args()

    mode = args.mode
    verbose = args.verbose

    if mode in ['backpropagation', 'gradient']:
        neural_network = load_model_from_text(args.network_path, args.weights_path)
        examples, test_results = load_dataset_from_text(args.dataset_path)

        ValidatorClass = GradientNumericValidator if mode == 'gradient' else BackpropagationValidator
        validator = ValidatorClass(neural_network, examples, test_results)
        validator.show(verbose=verbose)

    elif mode == 'train':
        data = load_dataset(args.dataset_path)
        normalized_data = normalize_dataset(data)
        attributes = get_attributes(normalized_data)
        expected = expected_to_neural_network(normalized_data)
        num_inputs = len(attributes.columns)
        num_outputs = len(expected.columns)
        layers = [num_inputs] + args.structure + [num_outputs]

        neural_network = NeuralNetwork(
            layers=layers,
            regularization_factor=args.regularization,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            show_loss=verbose)

        neural_network.train(attributes.values, expected.values)
        neural_network.save(args.output_path)

    elif mode == 'predict':
        data = load_dataset(args.dataset_path)
        normalized_data = normalize_dataset(data)
        attributes, expected, expected_columns = attributes_and_target(normalized_data)

        neural_network = NeuralNetwork.load(args.model_path)
        test_results = neural_network.predict(attributes.values)

        results_df = results_to_labels(test_results, expected_columns).join(expected)
        results_df.to_csv(args.results_path, index=False)

    elif mode == 'validate':
        data = load_dataset(args.dataset_path)
        normalized_data = normalize_dataset(data)

        k_folds = KFolds(normalized_data, args.k_folds, sampling='stratified')
        splits = k_folds.split_generator()

        all_results = []
        for train, test in splits:
            train_attributes, train_expected, train_columns = attributes_and_target(train)
            test_attributes, test_expected, test_columns = attributes_and_target(test)

            train_expected = expected_to_neural_network(train_expected)

            layers = generate_structure(train_attributes, train_expected, args.structure)

            neural_network = NeuralNetwork(
                layers=layers,
                regularization_factor=args.regularization,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                show_loss=verbose
            )

            neural_network.train(train_attributes.values, train_expected.values)
            test_results = neural_network.predict(test_attributes.values)
            results = results_to_labels(test_results, test_columns).join(test_expected)
            all_results.append(results)

        confusion_matrix = ConfusionMatrix(pd.concat(all_results))
        confusion_matrix.show()

    else:
        raise Exception('No valid execution mode was found. \
            Should be one of [Backpropagation, gradient, train, predict, validate]')


if __name__ == "__main__":
    main()
