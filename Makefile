VENV=.venv
# Prefer a compatible Python (3.11) when available, else fall back to python3
PYTHON := $(shell command -v python3.11 || command -v python3 || command -v python)
PY=${VENV}/bin/python
PIP=${VENV}/bin/pip

.PHONY: venv install run experiments test clean

venv:
	@echo "Using Python: ${PYTHON}"
	${PYTHON} -m venv ${VENV}
	${PIP} install --upgrade pip setuptools wheel

install: venv
	${PY} -m pip install --upgrade pip
	${PY} -m pip install -r requirements.txt
	# install pytest for local testing convenience
	${PY} -m pip install pytest

run:
	${PY} -m src.cli run

experiments:
	${PY} -m src.cli experiments

test:
	${PY} -m pytest -q

clean:
	rm -rf ${VENV} build dist *.egg-info
