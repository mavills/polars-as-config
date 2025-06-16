# Multiple Dataframes Feature

This document describes the multiple dataframes functionality added to the polars-as-config library, which allows working with multiple named dataframes simultaneously within a single configuration.

## Overview

The multiple dataframes feature enables users to:
- Create and manage multiple dataframes with unique identifiers
- Reference dataframes by name in operations that require multiple dataframes (e.g., joins, concatenations)
- Automatically resolve dataframe references through type-hint inspection
- Maintain clean, readable configuration syntax while supporting complex multi-dataframe operations

## Implementation Details

### Core Components

#### 1. Dataframe Tracking (`current_dataframes`)
The `Config` class maintains a dictionary `current_dataframes` that maps string identifiers to actual Polars DataFrame objects:

```python
self.current_dataframes: dict[str, pl.DataFrame] = {}
```

Each operation can specify a `dataframe` field in the configuration to indicate which named dataframe it should operate on or create.

#### 2. Type-Hint Detection (`is_dataframe`)
The system uses Python's `inspect` module to examine method signatures and detect parameters that expect DataFrame objects:

```python
def is_dataframe(self, key: str, type_hints: dict[str, Parameter]) -> bool:
    """Check if the key is a dataframe type hint."""
    if key not in type_hints:
        return False
    try:
        return eval(type_hints[key].annotation) is DataFrame
    except NameError:
        return False
```

This method:
- Checks if a parameter exists in the type hints
- Evaluates the type annotation to determine if it matches `pl.DataFrame`
- Returns `True` if the parameter expects a DataFrame, `False` otherwise

#### 3. Automatic Parameter Resolution (`parse_kwargs`)
When processing operation parameters, the system automatically substitutes dataframe references:

```python
def parse_kwargs(self, kwargs: dict, variables: dict, type_hints: dict):
    for key, value in kwargs.items():
        if isinstance(value, str):
            if self.is_dataframe(key, type_hints):
                if value not in self.current_dataframes:
                    raise ValueError(f"Dataframe {value} not found...")
                kwargs[key] = self.current_dataframes[value]
            # ... other string processing
```

This process:
- Identifies string parameters that should be dataframes (via type hints)
- Validates that the referenced dataframe exists
- Substitutes the actual DataFrame object for the string identifier
- Raises descriptive errors for missing dataframe references

#### 4. Step Processing (`handle_step`)
Each step in the configuration can specify which dataframe it operates on:

```python
def handle_step(self, current_data: Optional[pl.DataFrame], step: dict, variables: dict):
    # ... operation setup
    parameter_types = inspect.signature(method).parameters
    parsed_kwargs = self.parse_kwargs(kwargs, variables, type_hints=parameter_types)
    return method(*parsed_args, **parsed_kwargs)
```

The system:
- Extracts type information from the target method
- Uses this information to resolve dataframe parameters correctly
- Executes the operation with properly resolved parameters

#### 5. Configuration Execution (`run_config`)
The main execution loop processes steps and manages dataframe state:

```python
def run_config(self, config: dict):
    for step in steps:
        dataframe_name = step.get("dataframe", None)
        self.current_dataframes[dataframe_name] = self.handle_step(
            self.current_dataframes.get(dataframe_name), step, variables
        )
    return self.current_dataframes
```

This approach:
- Allows each step to specify which dataframe it targets
- Maintains separate dataframe states
- Returns all dataframes for potential further processing

## Configuration Syntax

### Basic Dataframe Operations
```json
{
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe": "customers",
      "kwargs": {"source": "customers.csv"}
    },
    {
      "operation": "scan_csv", 
      "dataframe": "orders",
      "kwargs": {"source": "orders.csv"}
    }
  ]
}
```

### Multi-Dataframe Operations
```json
{
  "steps": [
    {
      "operation": "join",
      "dataframe": "customers",
      "kwargs": {
        "other": "orders",
        "left_on": "customer_id",
        "right_on": "customer_id",
        "how": "inner"
      }
    }
  ]
}
```

In this example:
- The `join` operation is performed on the `"customers"` dataframe
- The `"other"` parameter references the `"orders"` dataframe by name
- The system automatically detects that `other` expects a DataFrame (via type hints)
- The string `"orders"` is replaced with the actual DataFrame object

## Benefits

### 1. Clean Configuration Syntax
- Dataframes are referenced by simple, meaningful names
- No need for complex object references or IDs
- Configuration remains human-readable and maintainable

### 2. Type Safety
- Automatic validation that dataframe references exist
- Type-hint based parameter resolution prevents type errors
- Clear error messages for missing or invalid references

### 3. Flexibility
- Support for arbitrary numbers of dataframes
- Operations can work with any combination of dataframes
- Easy to extend for new multi-dataframe operations

### 4. Backward Compatibility
- Existing single-dataframe configurations continue to work
- Default behavior (no `dataframe` field) uses `None` as identifier
- Gradual migration path for existing configurations

## Error Handling

The system provides clear error messages for common issues:

### Missing Dataframe Reference
```
ValueError: Dataframe 'orders' not found in current dataframes. 
It is possible that the dataframe was not created in the previous steps.
```

### Type Hint Resolution Errors
If type hint evaluation fails (e.g., due to missing imports), the system gracefully falls back to treating the parameter as a regular string.

## Usage Examples

### Example 1: Customer-Order Join
```json
{
  "variables": {
    "customer_file": "customers.csv",
    "orders_file": "orders.csv"
  },
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe": "customers", 
      "kwargs": {"source": "$customer_file"}
    },
    {
      "operation": "scan_csv",
      "dataframe": "orders",
      "kwargs": {"source": "$orders_file"}
    },
    {
      "operation": "join",
      "dataframe": "customers",
      "kwargs": {
        "other": "orders",
        "left_on": "id",
        "right_on": "customer_id"
      }
    }
  ]
}
```

### Example 2: Multiple Transformations and Concatenation
```json
{
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe": "sales_q1",
      "kwargs": {"source": "q1_sales.csv"}
    },
    {
      "operation": "scan_csv", 
      "dataframe": "sales_q2",
      "kwargs": {"source": "q2_sales.csv"}
    },
    {
      "operation": "with_columns",
      "dataframe": "sales_q1",
      "kwargs": {
        "quarter": {"expr": "lit", "kwargs": {"value": "Q1"}}
      }
    },
    {
      "operation": "with_columns",
      "dataframe": "sales_q2", 
      "kwargs": {
        "quarter": {"expr": "lit", "kwargs": {"value": "Q2"}}
      }
    },
    {
      "operation": "concat",
      "dataframe": "annual_sales",
      "kwargs": {
        "items": ["sales_q1", "sales_q2"]
      }
    }
  ]
}
```

This feature significantly enhances the flexibility and power of the polars-as-config library while maintaining its core principle of providing a clean, configuration-driven approach to data transformation. 