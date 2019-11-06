.PHONY: setup
setup:
	pip install -r requirements.txt

.PHONY: setup-dev
setup-dev: setup
	pip install -r requirements_dev.txt

.PHONY: lint
lint:
	flake8
