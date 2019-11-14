from src.command_line import get_args
from src.neural_network import NeuralNetwork
from src.file_system import load_dataset_from_text, load_network_from_text, load_weights_from_text
from src.numeric_validation import BackpropagationValidator


def main():
    args = get_args()

    mode = args.mode

    if mode == 'backpropagation':
        regularization, layers = load_network_from_text(args.network_path)
        examples, results = load_dataset_from_text(args.dataset_path)
        weights = load_weights_from_text(args.weights_path)

        neural_network = NeuralNetwork(layers=layers, weights=weights, regularization_factor=regularization)
        bp_validator = BackpropagationValidator(neural_network, examples, results)
        bp_validator.show_gradients()


if __name__ == "__main__":
    main()
