VENV=.venv
PY=${VENV}/bin/python
PIP=${VENV}/bin/pip

.PHONY: venv install run experiments test clean

venv:
	python3 -m venv ${VENV}
	${PIP} install --upgrade pip

install: venv
	${PIP} install -r requirements.txt

run:
	${PY} -m src.cli run

experiments:
	${PY} -m src.cli experiments

test:
	${PY} -m pytest -q

clean:
	rm -rf ${VENV} build dist *.egg-info
