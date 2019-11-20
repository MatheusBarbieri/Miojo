import pandas as pd
import numpy as np


class ConfusionMatrix:
    def __init__(self, results, predicted_col='predicted', expected_col='expected'):
        predicted = results[predicted_col]
        expected = results[expected_col]

        self._results = results
        self._total = len(results)
        self._correct = len(results[predicted == expected])
        self._cm = self._create_confusion_matrix(predicted, expected)

    def _create_confusion_matrix(self, expected, predicted):
        labels = np.sort(np.unique(np.append(predicted.unique(), expected.unique())))
        num_labels = len(labels)
        labels_index = {label: idx for idx, label in enumerate(labels)}
        new_matrix = np.zeros(shape=(num_labels, num_labels))

        for exp, pred in zip(expected, predicted):
            new_matrix[labels_index[exp]][labels_index[pred]] += 1

        return pd.DataFrame(
            new_matrix,
            index=labels,
            columns=labels
        )

    def __add__(self, val):
        new_results = pd.concat([self._results, val.results])
        return ConfusionMatrix(new_results)

    def __str__(self):
        return self._cm.__repr__()

    def print(self):
        print(self._cm)

    def true_positives(self):
        tps = pd.Series(np.diag(self._cm), index=self._cm.index)
        return tps.rename_axis('TruePositives')

    def true_negatives(self):
        tns = [self._cm.drop(index=[c], columns=[c]).values.sum() for c in self._cm.index]
        return pd.Series(tns, index=self._cm.index).rename_axis('TrueNegatives')

    def false_positives(self):
        fps = [self._cm.loc[c].sum() - self._cm[c][c] for c in self._cm.index]
        return pd.Series(fps, index=self._cm.index).rename_axis('FalsePositives')

    def false_negatives(self):
        fns = [self._cm[c].sum() - self._cm[c][c] for c in self._cm.index]
        return pd.Series(fns, index=self._cm.index).rename_axis('FalseNegatives')

    def accuracy(self):
        return self._correct / self._total

    def error(self):
        return 1 - self.accuracy()

    def recalls(self):
        tps = self.true_positives()
        fns = self.false_negatives()
        recalls = tps / (tps + fns)
        return recalls.rename_axis('Recall')

    def precisions(self):
        tps = self.true_positives()
        fps = self.false_positives()
        precisions = tps / (tps + fps)
        return precisions.rename_axis('Precision')

    def specificities(self):
        tns = self.true_negatives()
        fps = self.false_positives()
        specificities = tns / (tns + fps)
        return specificities.rename_axis('Specificity')

    def f_measures(self, b):
        prec = self.precisions()
        rec = self.recalls()
        f_measures = ((1 + b**2) * prec * rec / (b**2 * prec + rec))
        return f_measures.rename_axis('F-measure')

    def macro_recall(self):
        return self.recalls().mean()

    def macro_precision(self):
        return self.precisions().mean()

    def macro_specificity(self):
        return self.specificities().mean()

    def macro_f_measure(self, b):
        return self.f_measures(b).mean()

    def micro_recall(self):
        tps = self.true_positives().sum()
        fns = self.false_negatives().sum()
        recall = tps / (tps + fns)
        return recall

    def micro_precision(self):
        tps = self.true_positives().sum()
        fps = self.false_positives().sum()
        precision = tps / (tps + fps)
        return precision

    def micro_f_measure(self, b):
        prec = self.precisions().sum()
        rec = self.recalls().sum()
        f_measure = ((1 + b**2) * prec * rec / (b**2 * prec + rec))
        return f_measure

    def string_stats(self, verbose=False):
        string = ''
        if verbose:
            string += f"{'-'*50}\n{self}\n"
        string += f"{'-'*50}\n"
        string += f"Accuracy: {self.accuracy():.3f} [Total: {self._total}, Correct: {self._correct}]\n"
        string += f"Macro Recall: {self.macro_recall():.3f}\n"

        if verbose:
            for k, v in self.recalls().items():
                string += f"  Recall for class {k}: {v:.3f}\n"
        string += f"Macro Precision: {self.macro_precision():.3f}\n"
        if verbose:
            for k, v in self.precisions().items():
                string += f"  Precision for class {k}: {v:.3f}\n"
        string += f"Macro Specificity: {self.macro_specificity():.3f}"
        if verbose:
            for k, v in self.specificities().items():
                string += f"\n  Specificity for class {k}: {v:.3f}"
        for b in [2, 1, 0.5]:
            string += f"\nMacro F-measure (ß = {b}): {self.macro_f_measure(b):.3f}"
            if verbose:
                for k, v in self.f_measures(b).items():
                    string += f"\n  F-measure (ß = {b}) for class {k}: {v:.3f}"
        string += f"\n{'-'*50}"
        return string

    def show(self, verbose=False):
        print(self.string_stats(verbose))
