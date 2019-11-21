import os
import pandas as pd

from src.file_system import load_dataset
from src.neural_network import NeuralNetwork
from src.mathematics import normalize


def execute(args):
    data = load_dataset(args.dataset_path)
    normalized_data = normalize(data.values)

    neural_network = NeuralNetwork.load(args.model_path)
    results = neural_network.predict(normalized_data)

    print(f'Prediction finished, saving results to {args.results_path}')

    dirname = os.path.basename(args.results_path)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except Exception as e:
            print(f'Could not create {dirname} path to save model.')
            raise e

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.results_path, index=False)
