import argparse


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='Random Forest Classifier')

    parser.add_argument(
        '-m',
        '--mode',
        choices=['backpropagation', 'gradient', 'train', 'test', 'validate'],
        required=True,
        help="Mode to operate"
    )

    args = parser.parse_args(arguments)
    return args
