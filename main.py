import pandas as pd

from src.command_line import get_args
from src.file_system import load_model_from_text, load_dataset_from_text, load_dataset
from src.numeric_validation import BackpropagationValidator, GradientNumericValidator
from src.util import (
    normalize_dataset,
    get_attributes,
    attributes_and_target,
    results_to_labels,
    expected_to_neural_network
)
from src.neural_network import NeuralNetwork


def main():
    args = get_args()

    mode = args.mode
    verbose = args.verbose

    if mode in ['backpropagation', 'gradient']:
        neural_network = load_model_from_text(args.network_path, args.weights_path)
        examples, results = load_dataset_from_text(args.dataset_path)

        ValidatorClass = GradientNumericValidator if mode == 'gradient' else BackpropagationValidator
        validator = ValidatorClass(neural_network, examples, results)
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
        results = neural_network.predict(attributes.values)

        results_df = results_to_labels(results, expected_columns).join(expected)
        results_df.to_csv(args.results_path, index=False)


if __name__ == "__main__":
    main()
