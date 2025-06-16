# Transformation Final Format

This document describes the "Transformation Final Format," which is the fully resolved, executable configuration for a data transformation pipeline. This format is the result of evaluating or "compiling" a `transformation graph` that was constructed using `template blocks` (as defined in `transformation-template-format.md`).

The final transformation configuration is a JSON object that contains all the necessary information to execute a series of data manipulation operations, primarily using the Polars library.

## Core Concepts

- **Executable:** This format is designed to be directly interpretable by an execution engine.
- **Ordered Operations:** Operations are listed in the precise order they need to be executed.
- **Multiple Dataframe Management:** The system now supports tracking and managing multiple dataframes simultaneously. Each dataframe is identified by a unique name and can be referenced by subsequent operations through type-hint checking and automatic parameter substitution.
- **Type-Hint Aware Parameter Resolution:** The system uses Python type hints to automatically detect when a parameter expects a dataframe object and substitutes the actual dataframe instead of its string identifier.
- **Resolved Configuration:** All placeholders, connections, and parameters defined at the template or graph level have been resolved into concrete instructions.
- **Variables for Runtime:** It includes a consolidated list of variables required at runtime, typically for specifying input/output paths or other dynamic parameters.

## Multiple Dataframes Support

The system supports working with multiple dataframes simultaneously through the following mechanisms:

### Dataframe Tracking
- Each operation can specify a `dataframe` field to indicate which named dataframe it should operate on or create
- The system maintains a registry of all active dataframes using their string identifiers
- If no `dataframe` field is specified, operations default to using `None` as the dataframe identifier

### Type-Hint Based Parameter Resolution
- The system inspects the type hints of Polars operations to determine which parameters expect dataframe objects
- When a parameter is detected as expecting a `pl.DataFrame` type and the provided value is a string that matches an existing dataframe identifier, the system automatically substitutes the actual dataframe object
- This allows seamless referencing of dataframes by name in configuration while maintaining type safety

### Dataframe Parameter Syntax
- Dataframe references in configuration use simple string identifiers (e.g., `"customers"`, `"orders"`)
- The system automatically resolves these to actual dataframe objects when the operation signature indicates a dataframe parameter is expected
- This enables operations like joins, concatenations, and other multi-dataframe operations to be configured declaratively

## JSON Structure

The root of the transformation final format is a JSON object with the following top-level keys:

```json
{
  "name": "string",
  "description": "string",
  "variables": [
    /* Array of VariableInstance objects */
  ],
  "steps": [
    /* Array of Step objects - updated from operations */
  ],
  "outputs": [
    /* Array of TransformationOutput objects */
  ]
}
```

### 1. `name`

- **Type:** `string`
- **Description:** A human-readable name for this specific transformation configuration.
- **Example:** `"Customer Data Onboarding Q3 2024"`

### 2. `description`

- **Type:** `string`
- **Description:** A more detailed description of what this transformation configuration does.
- **Example:** `"Transforms raw customer CSVs, cleanses data, and joins with product information."`

### 3. `variables`

- **Type:** `Array<VariableInstance>`
- **Description:** An aggregated list of all variables required by this transformation configuration. These variables originate from the `template blocks` used in the transformation graph and are intended to be supplied at runtime (e.g., file paths, specific parameters).
- **`VariableInstance` Object Structure:**
  ```json
  {
    "key": "string", // Unique identifier for the variable (e.g., "source_customer_data_file")
    "name": "string", // Human-readable name (e.g., "Source Customer Data CSV")
    "description": "string" // Optional: A description of what this variable is for
    // "value": "<any>"    // Optional: A default or pre-filled value. If not provided, it MUST be supplied at runtime.
  }
  ```
  - **`key`**: This key is used by operations (e.g., `read_csv`) or by the `outputs` section to reference the actual value provided at runtime.

### 4. `operations`

- **Type:** `Array<OperationInstance>`
- **Description:** A list of operations to be executed in sequence. Each operation defines what action to take, on which dataframe(s) it acts, and what dataframe it produces.
- **`OperationInstance` Object Structure:**
  ```json
  {
    "operation": "string", // The name of the Polars function/method to execute (e.g., "scan_csv", "with_columns", "join", "filter").
    "dataframe": "string | null", // Optional: Name/identifier of the dataframe this operation targets. If null or omitted, defaults to None.
    "args": ["any"], // Optional: Positional arguments for the operation.
    "kwargs": {
      /* object */
    } // Keyword arguments for the `operation`.
    // Values can be literals, Expression Objects, dataframe references (automatically resolved via type hints),
    // or variable references.
  }
  ```
  - **`operation`**: The Polars operation to execute (e.g., `"scan_csv"`, `"with_columns"`, `"join"`).
  - **`dataframe`**: The identifier of the dataframe this operation should operate on or create. The system tracks all dataframes by these identifiers.
  - **`args`**: Positional arguments passed to the operation.
  - **`kwargs`**: 
    - The structure of `kwargs` largely mirrors the Polars API for the given operation.
    - Can contain nested `ExpressionObject` structures for complex transformations.
    - **Dataframe References**: When a parameter expects a dataframe (detected via type hints), you can reference other dataframes by their string identifier. The system automatically substitutes the actual dataframe object.
    - **Variable References**: Use `$variable_name` syntax to reference runtime variables.
    - **Escaped Dollar Signs**: Use `$$` to include a literal `$` character.

### 5. `outputs`

- **Type:** `Array<TransformationOutput>`
- **Description:** Defines the final outputs of the transformation pipeline. Each output maps a dataframe produced during the operations to a destination, typically specified by a runtime variable.
- **`TransformationOutput` Object Structure:**
  ```json
  {
    "name": "string", // A logical name for this output (e.g., "final_customer_report", "cleaned_sales_data")
    "dataframe_id": "string", // The identifier of the dataframe to be outputted.
    "destination_variable_key": "string" // The `key` of a variable (from the top-level `variables` list)
    // whose runtime value specifies the destination (e.g., an output file path).
    // "format_options": { /* object */ } // Optional: e.g., for CSV: {"delimiter": ",", "include_header": true}
    // for Parquet: {"compression": "snappy"}
  }
  ```

## Example with Multiple Dataframes

Here's an example showing multiple dataframes being created, processed, and joined:

```json
{
  "name": "Customer Order Processing with Multiple Dataframes",
  "description": "Loads customer and order data into separate dataframes, processes each independently, then joins them.",
  "variables": [
    {
      "key": "customer_csv_path",
      "name": "Customer CSV File Path",
      "description": "Path to the input CSV file containing customer data."
    },
    {
      "key": "orders_parquet_path", 
      "name": "Orders Parquet File Path",
      "description": "Path to the input Parquet file containing order data."
    },
    {
      "key": "output_joined_data_path",
      "name": "Output Parquet File Path", 
      "description": "Path where the final joined data will be saved."
    }
  ],
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe": "customers",
      "kwargs": {
        "source": "$customer_csv_path",
        "has_header": true,
        "separator": ","
      }
    },
    {
      "operation": "with_columns",
      "dataframe": "customers",
      "kwargs": {
        "email_clean": {
          "expr": "str.to_lowercase",
          "on": {
            "expr": "col",
            "kwargs": { "name": "email" }
          }
        }
      }
    },
    {
      "operation": "scan_parquet",
      "dataframe": "orders",
      "kwargs": {
        "source": "$orders_parquet_path"
      }
    },
    {
      "operation": "filter",
      "dataframe": "orders", 
      "kwargs": {
        "predicate": {
          "expr": "gt",
          "on": {
            "expr": "col",
            "kwargs": { "name": "order_amount" }
          },
          "kwargs": { "other": 0 }
        }
      }
    },
    {
      "operation": "join",
      "dataframe": "customers",
      "kwargs": {
        "other": "orders",
        "left_on": "customer_id",
        "right_on": "user_id", 
        "how": "inner"
      }
    }
  ],
  "outputs": [
    {
      "name": "final_joined_customer_orders",
      "dataframe_id": "customers",
      "destination_variable_key": "output_joined_data_path"
    }
  ]
}
```

In this example:
- Two separate dataframes are created: `"customers"` and `"orders"`
- Each dataframe is processed independently with its own operations
- The final `join` operation references the `"orders"` dataframe by name in the `other` parameter
- The system automatically detects that `other` expects a dataframe (via type hints) and substitutes the actual `orders` dataframe object
- The final result is stored back in the `"customers"` dataframe

## Type-Hint Based Resolution Details

The system uses Python's `inspect` module to examine the type hints of Polars operations:

1. **Parameter Inspection**: For each operation, the system inspects the method signature to identify parameters that expect `pl.DataFrame` objects
2. **String Detection**: When a parameter value is a string and the parameter is typed as expecting a dataframe, the system checks if a dataframe with that identifier exists
3. **Automatic Substitution**: If a matching dataframe is found, the string identifier is replaced with the actual dataframe object
4. **Error Handling**: If a dataframe reference is used but no matching dataframe exists, the system raises a descriptive error

This approach allows for clean, readable configuration while maintaining the flexibility and type safety of the underlying Polars operations.

## Relationship to Template Format

The `transformation-template-format.md` describes how individual, reusable `template blocks` are defined. Users combine instances of these templates into a `transformation graph`. The "Transformation Final Format" is the outcome of processing (compiling/evaluating) that graph.

During this "compilation":

- Operations from all template instances are sequenced correctly according to the graph's topology.
- `Input` and `output` connections between template instances are resolved. For example, an `input` in a template that expects a column name would be replaced by the actual column name produced by a preceding operation and referenced correctly in the `kwargs` of the current operation.
- Paths within template operations (e.g., `"operations.0.kwargs.parsed_date.on.on.kwargs.name"` from the template's `input.path`) are resolved into direct values or structures within the `kwargs` of an `OperationInstance` in this final format.
- `Variables` defined within individual templates are aggregated into the top-level `variables` list of this final format.
- The `ExpressionObject` structures are maintained but are fully resolved – they don't contain unresolved path references from template inputs but directly define the Polars expressions.

This final format is more verbose than the template definitions but is self-contained and ready for an engine to execute the transformation step-by-step.
