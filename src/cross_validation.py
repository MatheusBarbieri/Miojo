import pandas as pd
from numpy.random import randint


class KFolds:
    def __init__(self, data, k_folds=5, sampling='stratified'):
        self._data = data
        self._num_folds = k_folds
        self._sampling = sampling

    def _random_k_folds(self, data, add_remaining, seed):
        df = data
        fold_size = len(df) // self._num_folds

        folds = []
        for i in range(self._num_folds):
            sample = df.sample(n=fold_size, random_state=seed)
            df = df.drop(sample.index, errors='ignore')
            folds.append(sample)

        if add_remaining:
            for i in range(len(df)):
                folds[i] = pd.concat([folds[i], df.iloc[i:i+1]])

        return folds

    def _stratified_k_folds(self, add_remaining, seed):
        groups = self._data.groupby('class')
        folds_by_groups = [self._random_k_folds(g, add_remaining, seed) for c, g in groups]
        folds = [pd.concat(folds_by_groups[x][y] for x in range(len(folds_by_groups))) for y in range(self._num_folds)]
        return folds

    def _k_folds(self, add_remaining=True, seed=randint(10000)):
        if self._sampling == 'random':
            return self._random_k_folds(self.data, add_remaining, seed)
        elif self._sampling == 'stratified':
            return self._stratified_k_folds(add_remaining, seed)
        else:
            raise Exception("Sampling parameter must be one of [stratified, random]")

    def _splits(self, folds):
        for i, fold in enumerate(folds):
            train = pd.concat(folds[:i] + folds[i + 1:]).reset_index(drop=True)
            test = folds[i].reset_index(drop=True)
            yield train, test

    def split_generator(self):
        folds = self._k_folds()
        return self._splits(folds)
