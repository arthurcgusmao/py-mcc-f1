.PHONY: install test setup upload_test upload

default: test

install:
	pip install --upgrade .

test:
	PYTHONPATH=. pytest

setup:
	rm -rf ./dist ./build && python setup.py sdist bdist_wheel

upload_test:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload:
	twine upload dist/*
