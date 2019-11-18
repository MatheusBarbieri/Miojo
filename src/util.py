import numpy as np
import pandas as pd
from src.mathematics import normalize


def add_bias(instance):
    return np.insert(instance, 0, 1, axis=0)


def chunks(array, chunks_size):
    total = len(array)
    has_last_chunk = int(total % chunks_size != 0)
    iterations = total // chunks_size + has_last_chunk

    chunks = []
    for i in range(iterations):
        chunks.append(array[i * chunks_size: (i + 1) * chunks_size])

    return np.array(chunks)


def normalize_dataset(dataset, target='class'):
    data_without_target = dataset.drop([target], axis=1)
    data_with_class = dataset['class'] \
        .to_frame() \
        .join(pd.DataFrame(normalize(data_without_target.values), columns=data_without_target.columns))
    return data_with_class


def get_attributes(data):
    attributes = data.drop(['class'], axis=1)
    return attributes


def attributes_and_target(data):
    columns_names = np.sort(data['class'].unique())
    expected = data['class'].to_frame(name='expected')
    attributes = get_attributes(data)
    return attributes, expected, columns_names


def results_to_labels(results, labels):
    results_with_labels = pd.DataFrame(results, columns=labels).idxmax(axis=1)
    return results_with_labels.to_frame(name='predicted')


def expected_to_neural_network(data, target_column='expected'):
    encoded = pd.get_dummies(data[target_column])
    encoded = encoded.reindex(np.sort(encoded.columns), axis=1)
    return encoded


def generate_structure(attributes, expected, hidden_layers):
    num_inputs = len(attributes.columns)
    num_outputs = len(expected.columns)
    layers = [num_inputs] + hidden_layers + [num_outputs]
    return layers
