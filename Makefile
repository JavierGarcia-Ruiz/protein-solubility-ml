# Pipeline targets. On Windows/PowerShell run the equivalent `python scripts/...` commands
# listed in README.md ("How to Reproduce").
PY ?= python
CFG ?= configs/default.yaml

.PHONY: all features train figures test lint notebooks

all: features train figures test

features:
	$(PY) scripts/preprocess.py --config $(CFG)

train:
	$(PY) scripts/train.py --config $(CFG)

figures:
	$(PY) scripts/make_figures.py --config $(CFG)

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src scripts tests
