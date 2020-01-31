flake8:
	flake8 .

isort:
	isort -rc .

isort-check:
	isort -c -rc .

lint: flake8 isort-check

format: isort
