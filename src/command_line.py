import argparse


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='NeuralNetwork')
    help_str = 'Working mode, can be one of [backpropagation, gradient, train]'
    subparsers = parser.add_subparsers(dest='mode', help=help_str)
    subparsers.required = True

    # Backpropagation Validation
    parser_backpropagation = subparsers.add_parser('backpropagation', help='Backpropagation validation')
    parser_backpropagation.add_argument(
        "-n",
        "--network_path",
        type=str,
        required=True,
        help='Path to network file.'
    )

    parser_backpropagation.add_argument(
        "-w",
        "--weights_path",
        type=str,
        required=True,
        help='Path to weights file.'
    )

    parser_backpropagation.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help='Path to dataset file.'
    )

    parser_backpropagation.add_argument(
        '-v',
        '--verbose',
        action='count'
    )

    # Gradient Numeric Validation
    parser_gradient = subparsers.add_parser('gradient', help='Gradient numeric verification')
    parser_gradient.add_argument(
        "-n",
        "--network_path",
        type=str,
        required=True,
        help='Path to network file.'
    )

    parser_gradient.add_argument(
        "-w",
        "--weights_path",
        type=str,
        required=True,
        help='Path to weights file.'
    )

    parser_gradient.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help='Path to dataset file.'
    )

    parser_gradient.add_argument(
        '-v',
        '--verbose',
        action='count'
    )

    # Train
    parser_train = subparsers.add_parser('train', help='Neural Network traininig')
    parser_train.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help='Path to dataset file.'
    )

    parser_train.add_argument(
        "-t",
        "--target_classes",
        nargs='+',
        type=str,
        required=True,
        help='Name of dataset columns which are gonna be used as target classes/values.'
    )

    parser_train.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help='Path to where trained model weights file is gonna be output.'
    )

    parser_train.add_argument(
        "-h",
        "--hidden_layers",
        nargs='+',
        type=int,
        required=True,
        help='List of values that represents the number of neurons in inner layers'
    )

    parser_train.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.01,
        help='Learning rate of the model (weight update factor).'
    )

    parser_train.add_argument(
        "-r",
        "--regularization",
        type=float,
        default=0.1,
        help='Regularization factor parameter.'
    )

    parser_train.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help='Size of minibatches used.'
    )

    parser_train.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help='Size of minibatches used.'
    )

    parser_train.add_argument(
        '-v',
        '--verbose',
        action='count'
    )

    args = parser.parse_args(arguments)
    return args
