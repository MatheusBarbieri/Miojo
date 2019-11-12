import pandas as pd
from numpy.random import randint


def _random_k_folds(df, k, add_remaining, seed):
    fold_size = len(df) // k

    folds = []
    for i in range(k):
        sample = df.sample(n=fold_size, random_state=seed)
        df = df.drop(sample.index, errors='ignore')
        folds.append(sample)

    # add remaining elements to folds
    if add_remaining:
        for i in range(len(df)):
            folds[i] = pd.concat([folds[i], df.iloc[i:i+1]])

    return folds


def _stratified_k_folds(df, k, add_remaining, seed):
    groups = df.groupby('class')
    folds_by_groups = [_random_k_folds(g, k, add_remaining, seed) for c, g in groups]
    folds = [pd.concat(folds_by_groups[x][y] for x in range(len(folds_by_groups))) for y in range(k)]
    return folds


def k_folds(df, k, sampling='stratified', add_remaining=True, seed=randint(10000)):
    if sampling == 'random':
        return _random_k_folds(df, k, add_remaining, seed)
    elif sampling == 'stratified':
        return _stratified_k_folds(df, k, add_remaining, seed)
    else:
        raise Exception("Sampling parameter must be one of [stratified, random]")


def splits(folds):
    sets = []
    for i, fold in enumerate(folds):
        sets.append((pd.concat(folds[:i] + folds[i + 1:]), folds[i]))

    return sets
