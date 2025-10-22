# Install dependencies
install:
	python -m pip install -r requirements.txt

# Run tests
test:
	python -m pytest -v tests/

# Run pylint on all Python files in functions and tests
lint:
	python -m pylint --disable=R,C,E1120 functions/*.py tests/*.py

# Format code with black 
format:
	python -m black functions/*.py tests/*.py

# Run lint and format together
refactor: lint format

# Run all tasks
all: install lint test format