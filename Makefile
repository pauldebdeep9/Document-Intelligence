.PHONY: install update env-check test lint

PYTHON ?= python

install:
	conda env create -f environment.yml

update:
	conda env update -f environment.yml --prune

env-check:
	$(PYTHON) -V
	which python
	conda info --envs

test:    ; $(PYTHON) -m pytest tests -q
lint:    ; $(PYTHON) -m ruff check demo.py src/isc tests evals
	$(PYTHON) -m mypy src/isc demo.py evals
