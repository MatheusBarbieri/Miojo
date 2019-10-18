import argparse


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='Random Forest Classifier')

    # Add Args

    args = parser.parse_args(arguments)
    return args
