from src.file_system import load_dataset
from src.neural_network import NeuralNetwork
from src.util import (
    attributes_and_target,
    normalize_dataset,
    results_to_labels
)


def execute(args):
    data = load_dataset(args.dataset_path)
    normalized_data = normalize_dataset(data)
    attributes, expected, expected_columns = attributes_and_target(normalized_data)

    neural_network = NeuralNetwork.load(args.model_path)
    test_results = neural_network.predict(attributes.values)

    results_df = results_to_labels(test_results, expected_columns).join(expected)
    print(f'Prediction finished, saving results to {args.results_path}')
    results_df.to_csv(args.results_path, index=False)
