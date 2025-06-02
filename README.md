# Polars as Config

This library allows you to define Polars operations using a configuration format (JSON or Python dict), making it easy to serialize, store, and share data processing pipelines.

For a high-level overview, mission, vision, and a list of features, please see the [[Home|Home]] page of our Wiki.
For detailed technical explanations and examples of all features, please visit the [[Technical Features|Technical-Features]] page on our Wiki.

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

## Config Format Overview

The configuration is a JSON object (or Python dictionary) that describes a series of data processing steps. Each step in the `"steps"` array typically includes:

- `"operation"`: The Polars operation to perform (e.g., `"scan_csv"`, `"with_columns"`, `"filter"`).
- `"args"`: A list of positional arguments for the operation.
- `"kwargs"`: A dictionary of keyword arguments for the operation.

Complex operations and transformations within steps are defined using an **expression format**.

For a comprehensive guide on the config and expression formats, including various examples like basic operations, string operations, date operations, and advanced features like using variables and custom functions, please see our [[Technical Features|Technical-Features]] Wiki page.

## Expression Format Overview

Expressions allow you to define how data should be manipulated. Key components of an expression are:

- `"expr"`: The name of the Polars expression function (e.g., `"str.concat"`, `"eq"`, `"gt"`).
- `"on"`: The column or expression to apply this expression to (acting like `self` in an object-oriented context).
- `"args"` and `"kwargs"`: Positional and keyword arguments for the expression function.

**Example: `pl.col("x").gt(5)`**

```json
{
  "expr": "gt",
  "on": { "expr": "col", "kwargs": { "name": "x" } },
  "kwargs": { "other": 5 }
}
```

For more detailed examples and explanations of how to build simple and nested expressions, refer to the [[Technical Features|Technical-Features]] page on our Wiki.

## Installation

```bash
pip install polars-as-config
```

## Requirements

- Polars

## License

See [LICENSE](LICENSE) for details.
