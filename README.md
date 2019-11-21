# Miojo

> This is a neural network developed for the course: INF01017 - Aprendizado de Máquina (2019/2).

## Installation
This application requires Python version 3.6

There is a Makefile in the project folder which has rules for installing and running this application. Make sure you are on a python 3.6 environment and call `make setup`.

## Usage
The main file can be executed with one of the following commands:

- train
- predict
- backpropagation
- gradient
- evaluate-model
- evaluate-cost
- evaluate-parameters

>More datails on usage in the following sections

### Training Models and predicting

#### Train
Create and train a model for a given dataset, execution accepts the following parameters:

- **-d**: Path to dataset file. **Required**.
- **-o**: Path to where trained model/weights file is gonna be output. **Required**.
- **-s**: List of values that represents the number of neurons in inner layers. **Required**.
- **-l**: Learning rate of the model (weight update factor). Defaults to **0.01**.
- **-r**: Regularization factor parameter. Defaults to **0.1**.
- **-e**: Number of epochs used to train. Defaults to **100**.
- **-b**: Size of minibatches used. Defaults to **32**.
- **-v**: Verbosity.

Example execution:
```bash
python main.py train -d data/wine.csv -o models/wine.pickle -s 100 100 -r 0.01 -b 32 -l 0.001 -e 300 -v
```

After completion, a your model is gonna be ready for prediction.

#### Predict
Predict classes for examples and a given trained model, prediction accepts the following parameters:

- **-d**: Path to dataset file. **Required**.
- **-m**: Path to model file. **Required**.
- **-r**: Path to model file. **Required**.
- **-v**: Verbosity.

Example execution:
```bash
python main.py predict -d data/wine-predict.csv -m models/wine.pickle -r results/wine_results
```

A file that contains predicted probabilities is gonna be output. Each line represents the probabilities of each class for an input example.

### Evaluation

#### Evaluate-model
Evaluate performance of model for a given dataset and hyper parameters, model evaluation accepts the following parameters:

- **-d**: Path to dataset file. **Required**.
- **-s**: List of values that represents the number of neurons in inner layers. **Required**.
- **-l**: Learning rate of the model (weight update factor). Defaults to **0.01**.
- **-r**: Regularization factor parameter. Defaults to **0.1**.
- **-e**: Number of epochs used to train. Defaults to **100**.
- **-b**: Size of minibatches used. Defaults to **32**.
- **-k**: Number of k-folds used in model cross validation. Defaults to **5**.
- **-v**: Verbosity.

#### Evaluate-cost
Evaluate and save cost through the training of a model for a given dataset and parameters, cost evaluation accepts the following parameters:

- **-d**: Path to dataset file. **Required**.
- **-o**: Path to folder where results, metrics and stats are gonna be saved. **Required**.
- **-s**: List of values that represents the number of neurons in inner layers. **Required**.
- **-l**: Learning rate of the model (weight update factor). Defaults to **0.01**.
- **-r**: Regularization factor parameter. Defaults to **0.1**.
- **-e**: Number of epochs used to train. Defaults to **100**.
- **-b**: Size of minibatches used. Defaults to **32**.

#### Evaluate-parameters
Run about to 5000 different hyper parameter configurations for each given dataset, parameters evaluation accepts the following parameters:

- **-d**: List of paths of datasets that are gonna be evaluated. **Required**.
- **-o**: Path to folder where results, metrics and stats are gonna be saved. **Required**.
- **-k**: Number of k-folds used in model cross validation. Defaults to **5**.

#### Backpropagation
Validate backpropagation values, backpropagation validation accepts the following parameters:

- **-n**: Path to network file.
- **-w**: Path to weights file.
- **-d**: Path to dataset file.
- **-v**: Verbosity.

#### Gradient
Numeric validation of gradient values, gradient validation accepts the following parameters:

- **-n**: Path to network file.
- **-w**: Path to weights file.
- **-d**: Path to dataset file.
- **-v**: Verbosity.

For Backpropagation and Gradients validation, files must have the following format:
- Network file: first line is regularization factor, each following line represents number of neurons in a layer. Example:

```
0.25
2
4
3
2
```

- Weights file: each line represents weights connecting layer `i` to `i+1`, each weight group separated by `;` represents the weights for inputs of each neuron in the layer. Example:
```
0.42, 0.15, 0.40; 0.72, 0.10, 0.5; 0.01, 0.19, 0.42; 0.30, 0.35, 0.68
0.21, 0.67, 0.14, 0.96, 0.87; 0.87, 0.42, 0.20, 0.32, 0.89; 0.03, 0.56, 0.80, 0.69, 0.09
0.04, 0.87, 0.42, 0.53; 0.17, 0.10, 0.95, 0.69
```

- Dataset file: each line represents an example inputs and it's expected results, separated by `;`. Example:
```
0.32, 0.68; 0.75, 0.98
0.83, 0.02; 0.75, 0.28
```
