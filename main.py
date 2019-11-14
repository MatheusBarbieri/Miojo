from src.command_line import get_args
from src.file_system import load_model_from_text, load_dataset_from_text
from src.numeric_validation import BackpropagationValidator, GradientNumericValidator


def main():
    args = get_args()

    mode = args.mode

    if mode in ['backpropagation', 'gradient']:
        neural_network = load_model_from_text(args.network_path, args.weights_path)
        examples, results = load_dataset_from_text(args.dataset_path)

        ValidatorClass = GradientNumericValidator if mode == 'gradient' else BackpropagationValidator
        validator = ValidatorClass(neural_network, examples, results)
        validator.show()


if __name__ == "__main__":
    main()
