from src.command_line import get_args
from src import validation
from src import train
from src import predict
from src import evaluate_model
from src import evaluate_cost
from src.evaluate_parameters import ParametersEvaluationRunner


def main():
    args = get_args()
    mode = args.mode

    if mode in ['backpropagation', 'gradient']:
        validation.execute(args)

    elif mode == 'train':
        train.execute(args)

    elif mode == 'predict':
        predict.execute(args)

    elif mode == 'evaluate-model':
        evaluate_model.execute(args)

    elif mode == 'evaluate-cost':
        evaluate_cost.execute(args)

    elif mode == 'evaluate-parameters':
        ParametersEvaluationRunner(args).execute()


if __name__ == "__main__":
    main()
