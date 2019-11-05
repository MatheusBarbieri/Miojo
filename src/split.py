import pandas as pd


def generate_splits(folds):
    sets = []
    for i, fold in enumerate(folds):
        sets.append((pd.concat(folds[:i] + folds[i + 1:]), folds[i]))

    return sets
