# Repository context

This repository is a python library that aids in transforming arbitrary csv files to different formats using polars building blocks, but as configuration.

## Key Features

- **Configuration-driven transformations**: Define Polars operations using JSON or Python dictionaries
- **Multiple dataframes support**: Work with multiple named dataframes simultaneously within a single configuration
- **Type-hint based parameter resolution**: Automatic detection and substitution of dataframe references using Python type hints
- **Expression system**: Support for complex Polars expressions defined declaratively
- **Variable substitution**: Runtime variables for dynamic configuration (file paths, parameters)
- **Custom functions**: Extensible system for adding custom transformation functions

## Technology choices

- python 3.13
- polars (for python)
- we use pytest for testing
- The package is hosted on PyPi.
- `python -m build` for building the package
- `twine` for uploading to PyPi.
- the package is semver'd.
- I use `mise` to manage my virtual environments

## Preferences

- I prefer readable code above all else (simple, verbose, linear, no complex language features)
- I prefer everything to be automated and documented
- I prefer tests to be fast