from src.neural_network import NeuralNetwork
from src.file_system import load_dataset
from src.util import (
    expected_to_neural_network,
    get_attributes,
    normalize_dataset,
)


def execute(args):
    data = load_dataset(args.dataset_path)
    normalized_data = normalize_dataset(data)
    attributes = get_attributes(normalized_data)
    expected = expected_to_neural_network(normalized_data, target_column='class')
    num_inputs = len(attributes.columns)
    num_outputs = len(expected.columns)
    layers = [num_inputs] + args.structure + [num_outputs]

    neural_network = NeuralNetwork(
        layers=layers,
        regularization_factor=args.regularization,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbosity=args.verbose)

    neural_network.train(attributes.values, expected.values)
    print(f'Finished training! Saving model to {args.output_path}')
    neural_network.save(args.output_path)
