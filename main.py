from src.command_line import get_args
from src.file_system import load_model_from_text, load_dataset_from_text
from src.numeric_validation import BackpropagationValidator, GradientNumericValidator
from src import train
from src import predict
from src import validate


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
        train.execute(args)

    elif mode == 'predict':
        predict.execute(args)

    elif mode == 'validate':
        validate.execute(args)


if __name__ == "__main__":
    main()
