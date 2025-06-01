# Transformation Final Format

This document describes the "Transformation Final Format," which is the fully resolved, executable configuration for a data transformation pipeline. This format is the result of evaluating or "compiling" a `transformation graph` that was constructed using `template blocks` (as defined in `transformation-template-format.md`).

The final transformation configuration is a JSON object that contains all the necessary information to execute a series of data manipulation operations, primarily using the Polars library.

## Core Concepts

- **Executable:** This format is designed to be directly interpretable by an execution engine.
- **Ordered Operations:** Operations are listed in the precise order they need to be executed.
- **Dataframe Management:** The flow of data through different states (dataframes) is explicitly managed. Each operation acts on one or more input dataframes and produces an output dataframe.
- **Resolved Configuration:** All placeholders, connections, and parameters defined at the template or graph level have been resolved into concrete instructions.
- **Variables for Runtime:** It includes a consolidated list of variables required at runtime, typically for specifying input/output paths or other dynamic parameters.

## JSON Structure

The root of the transformation final format is a JSON object with the following top-level keys:

```json
{
  "name": "string",
  "description": "string",
  "variables": [
    /* Array of VariableInstance objects */
  ],
  "operations": [
    /* Array of OperationInstance objects */
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
    "id": "string", // A unique identifier for this operation instance within the transformation.
    "description": "string", // Optional: Human-readable description of this specific step.
    "polars_operation": "string", // The name of the Polars function/method to execute (e.g., "read_csv", "with_columns", "join", "filter").
    "input_dataframe_id": "string | null", // ID of the primary input dataframe. `null` if the operation creates the first dataframe (e.g., read_csv).
    "kwargs": {
      /* object */
    }, // Keyword arguments for the `polars_operation`.
    // Values can be literals, Expression Objects (similar to template format but fully resolved),
    // or references to other dataframes (e.g., for joins).
    "output_dataframe_id": "string" // ID assigned to the dataframe resulting from this operation. This ID can be used by subsequent operations.
  }
  ```
  - **`id`**: Useful for logging, debugging, and potentially for internal referencing if needed.
  - **`input_dataframe_id`**: Specifies the context. For an operation like `df.with_columns(...)`, this would be the ID of `df`. For `pl.read_csv(...)`, this would be `null`.
  - **`kwargs`**:
    - The structure of `kwargs` largely mirrors the Polars API for the given `polars_operation`.
    - It can contain nested `ExpressionObject` structures similar to those in `transformation-template-format.md` but fully resolved (no unresolved template `input` paths).
    - For operations requiring multiple dataframes (e.g., `join`, `concat`), `kwargs` will contain references to other `output_dataframe_id`s from previous steps. For example, a `join` might have `{"other_dataframe_id": "some_other_df_id", "left_on": "colA", "right_on": "colB"}`.
    - To reference a runtime variable (e.g., for a file path in `read_csv`), a special object can be used: `{"source": {"type": "variable_reference", "key": "my_input_file_var"}}`.
  - **`output_dataframe_id`**: Every operation produces a dataframe (even if it's conceptually an in-place modification, it results in a new state). This ID must be unique within the transformation.

### 5. `outputs`

- **Type:** `Array<TransformationOutput>`
- **Description:** Defines the final outputs of the transformation pipeline. Each output maps a dataframe produced during the operations to a destination, typically specified by a runtime variable.
- **`TransformationOutput` Object Structure:**
  ```json
  {
    "name": "string", // A logical name for this output (e.g., "final_customer_report", "cleaned_sales_data")
    "dataframe_id": "string", // The `output_dataframe_id` from one of the operations that represents the data to be outputted.
    "destination_variable_key": "string" // The `key` of a variable (from the top-level `variables` list)
    // whose runtime value specifies the destination (e.g., an output file path).
    // "format_options": { /* object */ } // Optional: e.g., for CSV: {"delimiter": ",", "include_header": true}
    // for Parquet: {"compression": "snappy"}
  }
  ```

## Example Snippet

Here's how a sequence of operations and outputs might look in a complete transformation JSON:

```json
{
  "name": "Customer Order Processing",
  "description": "Loads customer and order data, cleans customer emails, joins them, and outputs the result.",
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
      "description": "Path where the final joined data will be saved in Parquet format."
    }
  ],
  "operations": [
    {
      "id": "op_read_customers",
      "description": "Load customer data from CSV",
      "polars_operation": "read_csv",
      "input_dataframe_id": null,
      "kwargs": {
        "source": {
          "type": "variable_reference",
          "key": "customer_csv_path"
        },
        "has_header": true,
        "separator": ","
      },
      "output_dataframe_id": "df_customers_raw"
    },
    {
      "id": "op_clean_email",
      "description": "Lowercase email addresses in customer data",
      "polars_operation": "with_columns",
      "input_dataframe_id": "df_customers_raw",
      "kwargs": {
        "email": {
          "expr": "str.to_lowercase",
          "on": {
            "expr": "col",
            "kwargs": { "name": "email" }
          }
        }
      },
      "output_dataframe_id": "df_customers_cleaned_email"
    },
    {
      "id": "op_read_orders",
      "description": "Load order data from Parquet",
      "polars_operation": "read_parquet",
      "input_dataframe_id": null,
      "kwargs": {
        "source": {
          "type": "variable_reference",
          "key": "orders_parquet_path"
        }
      },
      "output_dataframe_id": "df_orders_raw"
    },
    {
      "id": "op_join_customer_orders",
      "description": "Join cleaned customer data with their orders",
      "polars_operation": "join",
      "input_dataframe_id": "df_customers_cleaned_email",
      "kwargs": {
        "other": {
          "type": "dataframe_reference",
          "id": "df_orders_raw"
        },
        "left_on": "customer_id",
        "right_on": "user_id",
        "how": "inner"
      },
      "output_dataframe_id": "df_customer_orders_joined"
    }
  ],
  "outputs": [
    {
      "name": "final_joined_customer_orders",
      "dataframe_id": "df_customer_orders_joined",
      "destination_variable_key": "output_joined_data_path"
      // "format_options": {"file_type": "parquet", "compression": "snappy"} // Example for specific output formatting
    }
  ]
}
```

## Relationship to Template Format

The `transformation-template-format.md` describes how individual, reusable `template blocks` are defined. Users combine instances of these templates into a `transformation graph`. The "Transformation Final Format" is the outcome of processing (compiling/evaluating) that graph.

During this "compilation":

- Operations from all template instances are sequenced correctly according to the graph's topology.
- `Input` and `output` connections between template instances are resolved. For example, an `input` in a template that expects a column name would be replaced by the actual column name produced by a preceding operation and referenced correctly in the `kwargs` of the current operation.
- Paths within template operations (e.g., `"operations.0.kwargs.parsed_date.on.on.kwargs.name"` from the template's `input.path`) are resolved into direct values or structures within the `kwargs` of an `OperationInstance` in this final format.
- `Variables` defined within individual templates are aggregated into the top-level `variables` list of this final format.
- The `ExpressionObject` structures are maintained but are fully resolved – they don't contain unresolved path references from template inputs but directly define the Polars expressions.

This final format is more verbose than the template definitions but is self-contained and ready for an engine to execute the transformation step-by-step.
