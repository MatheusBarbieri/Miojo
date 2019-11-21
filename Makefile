.PHONY: setup
setup:
	pip install -r requirements.txt

.PHONY: setup-dev
setup-dev: setup
	pip install -r requirements_dev.txt

.PHONY: lint
lint:
	flake8

.PHONY: evaluate-parameters
evaluate-parameters:
	python main.py evaluate-parameters -d data/breastCancer.csv data/diabetes.csv data/wine.csv data/ionosphere.csv -o results/
