import argparse


def add_verbosity(parser):
    parser.add_argument(
        '-v',
        '--verbose',
        action='count'
    )


def add_network_path(parser):
    parser.add_argument(
        "-n",
        "--network_path",
        type=str,
        required=True,
        help='Path to network file.'
    )


def add_weights_path(parser):
    parser.add_argument(
        "-w",
        "--weights_path",
        type=str,
        required=True,
        help='Path to weights file.'
    )


def add_dataset_path(parser):
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help='Path to dataset file.'
    )


def add_output_path(parser):
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help='Path to where trained model weights file is gonna be output.'
    )


def add_outputs_path(parser):
    parser.add_argument(
        "-o",
        "--outputs_path",
        type=str,
        required=True,
        help='Path to where validation results and metrics are gonna be saved.'
    )


def add_structure(parser):
    parser.add_argument(
        "-s",
        "--structure",
        nargs='+',
        type=int,
        required=True,
        help='List of values that represents the number of neurons in inner layers'
    )


def add_learning_rate(parser):
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.01,
        help='Learning rate of the model (weight update factor).'
    )


def add_regularization(parser):
    parser.add_argument(
        "-r",
        "--regularization",
        type=float,
        default=0.1,
        help='Regularization factor parameter.'
    )


def add_model_path(parser):
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help='Path to model file.'
    )


def add_results_path(parser):
    parser.add_argument(
        "-r",
        "--results_path",
        type=str,
        required=True,
        help='Path to model file.'
    )


def add_epochs(parser):
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help='Size of minibatches used.'
    )


def add_batch_size(parser):
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help='Size of minibatches used.'
    )


def add_k_folds(parser):
    parser.add_argument(
        "-k",
        "--k_folds",
        type=int,
        default=5,
        help='Number of k-folds used in model cross validation.'
    )


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='NeuralNetwork')
    help_str = 'Working mode, can be one of [backpropagation, gradient, train]'
    subparsers = parser.add_subparsers(dest='mode', help=help_str)
    subparsers.required = True

    # Backpropagation Validation
    parser_backpropagation = subparsers.add_parser('backpropagation', help='Backpropagation validation')
    add_network_path(parser_backpropagation)
    add_weights_path(parser_backpropagation)
    add_dataset_path(parser_backpropagation)
    add_verbosity(parser_backpropagation)

    # Gradient Numeric Validation
    parser_gradient = subparsers.add_parser('gradient', help='Gradient numeric verification')
    add_network_path(parser_gradient)
    add_weights_path(parser_gradient)
    add_dataset_path(parser_gradient)
    add_verbosity(parser_gradient)

    # Train
    parser_train = subparsers.add_parser('train', help='Neural Network traininig')
    add_dataset_path(parser_train)
    add_output_path(parser_train)
    add_structure(parser_train)
    add_learning_rate(parser_train)
    add_regularization(parser_train)
    add_epochs(parser_train)
    add_batch_size(parser_train)
    add_verbosity(parser_train)

    # Predict
    parser_predict = subparsers.add_parser('predict', help='Neural Network Prediction')
    add_dataset_path(parser_predict)
    add_epochs(parser_predict)
    add_batch_size(parser_predict)
    add_verbosity(parser_predict)

    # Validate
    parser_validate = subparsers.add_parser('validate', help='Neural Network Validation')
    add_dataset_path(parser_validate)
    add_outputs_path(parser_validate)
    add_structure(parser_validate)
    add_learning_rate(parser_validate)
    add_regularization(parser_validate)
    add_epochs(parser_validate)
    add_batch_size(parser_validate)
    add_k_folds(parser_validate)
    add_verbosity(parser_validate)

    args = parser.parse_args(arguments)
    return args
