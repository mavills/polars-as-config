# Polars as Config

This library allows you to define Polars operations using a configuration format (JSON or Python dict), making it easy to serialize, store, and share data processing pipelines.

## Quick Start

```python
from polars_as_config.config import run_config

# Define your operations in a config
config = {
    "steps": [
        # Read a CSV file
        {"operation": "scan_csv", "kwargs": {"source": "data.csv"}},

        # Add a new column by joining two string columns
        {
            "operation": "with_columns",
            "kwargs": {
                "full_name": {
                    "expr": "str.concat",
                    "on": {"expr": "col", "kwargs": {"name": "first_name"}},
                    "kwargs": {
                        "delimiter": " ",
                        "other": {"expr": "col", "kwargs": {"name": "last_name"}}
                    }
                }
            }
        }
    ]
}

# Run the config
result = run_config(config)
```

## Config Format

The config describes operations by defining the exact function to execute. Each step in the config:

1. Defines its operation in the "operation" key
2. Provides arguments in the "kwargs" key
3. Can use expressions to define complex operations

### Basic Operations

```python
# Reading a CSV file
config = {
    "steps": [
        {"operation": "scan_csv", "kwargs": {"source": "data.csv"}}
    ]
}

# Filtering rows
config = {
    "steps": [
        {
            "operation": "filter",
            "kwargs": {
                "predicate": {
                    "expr": "gt",
                    "on": {"expr": "col", "kwargs": {"name": "age"}},
                    "kwargs": {"other": 18}
                }
            }
        }
    ]
}
```

### String Operations

```python
# String concatenation
config = {
    "steps": [
        {
            "operation": "with_columns",
            "kwargs": {
                "full_name": {
                    "expr": "str.concat",
                    "on": {"expr": "col", "kwargs": {"name": "first"}},
                    "kwargs": {
                        "delimiter": "-",
                        "other": {"expr": "col", "kwargs": {"name": "last"}}
                    }
                }
            }
        }
    ]
}

# String slicing
config = {
    "steps": [
        {
            "operation": "with_columns",
            "kwargs": {
                "sliced": {
                    "expr": "str.slice",
                    "on": {"expr": "col", "kwargs": {"name": "text"}},
                    "kwargs": {
                        "offset": 1,
                        "length": 2
                    }
                }
            }
        }
    ]
}
```

### Date Operations

```python
# Converting strings to datetime
config = {
    "steps": [
        {
            "operation": "with_columns",
            "kwargs": {
                "parsed_date": {
                    "expr": "str.to_datetime",
                    "on": {"expr": "col", "kwargs": {"name": "date_str"}},
                    "kwargs": {
                        "format": "%Y-%m-%d %H:%M%#z"
                    }
                }
            }
        }
    ]
}
```

## Expression Format

Expressions are defined using three keys:

1. `expr`: The name of the expression function (e.g., "str.concat", "eq", "gt")
2. `on`: The expression to apply the operation to (like "self" in Python)
3. `kwargs`: Arguments for the expression

```python
# Example: x > 5 in polars: pl.col("x").gt(5)
{
    "expr": "gt",
    "on": {"expr": "col", "kwargs": {"name": "x"}},
    "kwargs": {"other": 5}
}

# Example: str1 + "-" + str2 in polars: pl.col("str1").str.concat("-", pl.col("str2"))
{
    "expr": "str.concat",
    "on": {"expr": "col", "kwargs": {"name": "str1"}},
    "kwargs": {
        "delimiter": "-",
        "other": {"expr": "col", "kwargs": {"name": "str2"}}
    }
}
```

## Installation

```bash
pip install polars-as-config
```

## Requirements

- Polars

## License

See [LICENSE](LICENSE) for details.
