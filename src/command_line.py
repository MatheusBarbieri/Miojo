import argparse


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='NeuralNetwork')

    help_str = 'Working mode, can be onde of [backpropagation, gradient]'
    subparsers = parser.add_subparsers(dest='mode', help=help_str)
    subparsers.required = True

    parser_backpropagation = subparsers.add_parser('backpropagation', help='Backpropagation validation')
    parser_backpropagation.add_argument("-n", "--network_path", type=str, required=True, help='Path to network file.')
    parser_backpropagation.add_argument("-w", "--weights_path", type=str, required=True, help='Path to weights file.')
    parser_backpropagation.add_argument("-d", "--dataset_path", type=str, required=True, help='Path to dataset file.')
    parser_backpropagation.add_argument('-v', '--verbose', action='count')

    parser_gradient = subparsers.add_parser('gradient', help='Gradient numeric verification')
    parser_gradient.add_argument("-n", "--network_path", type=str, required=True, help='Path to network file.')
    parser_gradient.add_argument("-w", "--weights_path", type=str, required=True, help='Path to weights file.')
    parser_gradient.add_argument("-d", "--dataset_path", type=str, required=True, help='Path to dataset file.')
    parser_gradient.add_argument('-v', '--verbose', action='count')

    args = parser.parse_args(arguments)
    return args
