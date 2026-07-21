# Contributing

## Setting up a development environment

Clone the repository and install all dependency groups:

```bash
git clone https://github.com/dgegen/astrafocus.git
cd astrafocus
uv sync --group dev --group docs --group test
```

Install the pre-commit hooks so linting runs automatically before each commit:

```bash
uv run pre-commit install
```

## Running the tests

```bash
uv run python -m pytest
```

## Linting and formatting

```bash
uv run ruff check .
uv run ruff format .
```

Or run all pre-commit hooks manually:

```bash
uv run pre-commit run --all-files
```

## Building the docs

```bash
cd docs
uv run make html
```

The output is written to `docs/build/html/`. Open `docs/build/html/index.html` to view it locally.
