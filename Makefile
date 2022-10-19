AUTOFLAKE_OPTIONS = --remove-all-unused-imports --remove-unused-variables 

clean:
	find . -name "__pycache__" | xargs rm -rf
	rm -rf poetry.lock .vscode .ipynb_checkpoints

format:
	set -x
	isort --force-single-line-imports .
	autoflake $(AUTOFLAKE_OPTIONS) --in-place --recursive .
	black .
	isort .
	black .

.PHONY: clean format