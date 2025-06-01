# Transformation template

## Concepts

The first part is creating transformations.
To create a transformation, you need building blocks that allow you to mix and combine, operate on, and select columns of a specific input file.
Since building blocks stand alone, they need to be able to interact with other building blocks.
This means that one of the requirements of a blocks is that is has inputs and outputs defined.
It also needs operations that it will execute, and some way of identifying and describing it.
Lastly, since the final goal is operating on a combined set of template blocks, there is some room for "variables", which are passed from the outside after transforming. These cannot be connected to inside a configuration, but only when the transformation is applied onto a set of input files to create output files (both specified as variables to the system!).

Defining these blocks is fully up to the implementer of them.
Once they are defined, they can simply be re-used, without having to know how inputs are connected to the operations: as long as it is clear what the input and output mean, the operations will be clear for an end-user.

## Operations

Operations are defined inside of the template block under the `operations` key, which represents a list of operations.
A single operations has the identifier of what it will do, which corresponds to the polars function you will call on a dataframe if you would be writing polars code.
So for example, if you would write `pl.read_csv(...).with_columns(...).join(...)`, the list of operations would be of length 3, where the operations are `read_csv`, `with_columns`, and `join`.

An operation also has a `kwargs` key. This key describes the keyword arguments you would like to pass on to the operation you are executing. How you fill those in is fully up to you, and how configurable you want your block template.
`kwargs` is an object, where the keys are the names of the keyword arguments (note that you cannot pass unnamed arguments), and the values is what will be passed along.

Values can be any standard type allowed in JSON, e.g. string, bool, integer, float.

There is currently one special type, the `expr` object. It represents expressions from polars, and you set a keyword argument to it by passing an object as a value with a key called `expr`.
Such an `expr` object should then define as the value of the `expr` key what the expression is.

You can define an `on` key, which then defines onto which other object (most likely another expression) the expression will operate.
If no `on` key is specified, the operation of the expression (e.g. `col`) will simply be executed on the main polars module instead of an object, which is `polars` or `pl` in the Python implementation.

An example of a single item in the `operations` array of the template block would be:

```json
{
  "operation": "with_columns",
  "kwargs": {
    "parsed_date": {
      "expr": "alias",
      "on": {
        "expr": "str.to_datetime",
        "on": {
          "expr": "col",
          "kwargs": {
            "name": "date_str"
          }
        },
        "kwargs": {
          "format": "%Y-%m-%d %H:%M%#z"
        }
      },
      "kwargs": {
        "name": "parsed_date"
      }
    }
  }
}
```

## Inputs and outputs

Inputs must be able to inject context into the operations a block will execute. This means that we must say where a specific input will operate. Since we define operations as paths, we can index the JSON strucure's operations with the path at which the input should be injected.
An example of this would be the following:

```JSON
# operations
{
  ...,
  "operations": [
    {
      "operation": "with_columns",
      "kwargs": {
        "parsed_date": {
          "expr": "alias",
          "on": {
            "expr": "str.to_datetime",
            "on": {
              "expr": "col",
              "kwargs": {
                "name": "date_str"
              }
            },
            "kwargs": {
              "format": "%Y-%m-%d %H:%M%#z"
            }
          },
          "kwargs": {
            "name": "parsed_date"
          }
        }
      }
    }
  ]
}
```

The above is the operation configuration for a date conversion template block.

An input could be defined as follows in that case:

```json
# input
{
  "name": "Date string", # Refers to what the input is, not a specific input
  "type": "column", # or any other allowed type in the block system
  "path": "operations.0.kwargs.parsed_date.on.on.kwargs.name", # follow this path to see that it operates on the input of the datetime conversion
  ...
}
```

Lastly, an input definition has a `default` field. The default determines what should happen when nothing is connected or defined for that input.

An output follows a similar scheme, but probably would operate on the polars "alias" operator's argument most of the time, since it determines the output name of the operation. Other operations will have other output paths.

When evaluating the output later on, for example for newly created columns, the output column should just be unique, no real other constraints. That ensures that if it is used as an input, the operator that takes it as input knows on which column to operate.
Another constraint we can add that is simply nice, is that we can start the output column name with the name of the operation and the id of the operation.
This makes debugging a whole lot easier, as we now know where the output came from.
Adding identifiers happens when a template block is instantiated during transformation configuration creation, so outside of the scope of this document.

## Variables

Variables are defined as a list of objects, and define their display names and how they are used in the template:

```json
{
  ...,
  "variables": [
    {
      "name": "Weather data file name",
      "key": "weather_data_file"
    },
    {
      "name": "Location file name",
      "key": "location_file"
    },
    {...},
    ...
  ]
}
```

They can be used in the configuration by referencing the `key` using a dollar sign `$`.
So any string that starts with a dollar sign in the template will be interpreted as a variable and evaluated as such (by simple replacement at transformation time).
If for some reason you would need a dollar sign in the definition, I propose to use two dollar signs `$$`.
I will refrain from implementing this for now, as I see no practical use-cases for dollar signs.

## API reference of transformation template

### Template definition

A transformation template is a JSON object that defines a reusable building block for data transformations.

**Structure:**

```json
{
  "name": "<string>",
  "description": "<string>",
  "inputs": [<InputDefinition>, ...],
  "outputs": [<OutputDefinition>, ...],
  "operations": [<OperationDefinition>, ...],
  "variables": [<VariableDefinition>, ...]
}
```

**Fields:**

| Field         | Type                    | Required | Description                                                                                            |
| ------------- | ----------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `name`        | `string`                | Yes      | Unique identifier for the template. Used for referencing and instantiation.                            |
| `description` | `string`                | Yes      | Human-readable description of what this template does and when to use it.                              |
| `inputs`      | `InputDefinition[]`     | No       | Array of input definitions that describe what data this template expects. Empty array if no inputs.    |
| `outputs`     | `OutputDefinition[]`    | No       | Array of output definitions that describe what data this template produces. Empty array if no outputs. |
| `operations`  | `OperationDefinition[]` | Yes      | Array of operations that will be executed. Must contain at least one operation.                        |
| `variables`   | `VariableDefinition[]`  | No       | Array of variable definitions. These are placeholders for values provided at transformation runtime.   |

### Input definition

Defines an input parameter that can be configured when using the template.

**Structure:**

```json
{
  "name": "<string>",
  "type": "<string>",
  "path": "<string>",
  "default": <any>
}
```

**Fields:**

| Field     | Type     | Required | Description                                                                                                                                                                    |
| --------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`    | `string` | Yes      | Human-readable name describing what this input represents (e.g., "Date string", "Source column").                                                                              |
| `type`    | `string` | Yes      | Type of input expected. Valid values: `"column"`, `"value"`, `"expression"`, `"template"`.                                                                                     |
| `path`    | `string` | Yes      | JSONPath-style string indicating where this input value should be injected in the operations. Uses dot notation (e.g., `"operations.0.kwargs.parsed_date.on.on.kwargs.name"`). |
| `default` | `any`    | Yes      | Default value to use when no input is provided. Type should match the expected input type.                                                                                     |

**Input Types:**

- `"column"`: Expects a column name (string)
- `"value"`: Expects a literal value (string, number, boolean)
- `"expression"`: Expects a polars expression object
- `"template"`: Expects another template instance

### Output definition

Defines an output that this template produces.

**Structure:**

```json
{
  "name": "<string>",
  "type": "<string>",
  "path": "<string>"
}
```

**Fields:**

| Field  | Type     | Required | Description                                                                                                                                      |
| ------ | -------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name` | `string` | Yes      | Human-readable name describing what this output represents (e.g., "Parsed date", "Filtered data").                                               |
| `type` | `string` | Yes      | Type of output produced. Valid values: `"column"`, `"dataframe"`, `"value"`.                                                                     |
| `path` | `string` | Yes      | JSONPath-style string indicating where the output value is defined in the operations. Typically points to an `alias` operation's name parameter. |

**Output Types:**

- `"column"`: Produces a new column
- `"dataframe"`: Produces a transformed dataframe
- `"value"`: Produces a computed value

### Variable definition

Defines a variable that can be set when the transformation is applied. These variables are typically used to provide context-specific values like file paths or runtime parameters.

**Structure:**

```json
{
  "name": "<string>",
  "key": "<string>"
}
```

**Fields:**

| Field  | Type     | Required | Description                                                                              |
| ------ | -------- | -------- | ---------------------------------------------------------------------------------------- |
| `name` | `string` | Yes      | Human-readable name for the variable, used for display or documentation purposes.        |
| `key`  | `string` | Yes      | Unique key used to identify and reference the variable when applying the transformation. |

### Operation definition

Defines a single operation to be executed, corresponding to a polars dataframe method.

**Structure:**

```json
{
  "operation": "<string>",
  "kwargs": {
    "<param_name>": <value | ExpressionObject>,
    ...
  }
}
```

**Fields:**

| Field       | Type     | Required | Description                                                                                                         |
| ----------- | -------- | -------- | ------------------------------------------------------------------------------------------------------------------- |
| `operation` | `string` | Yes      | Name of the polars operation to execute (e.g., `"with_columns"`, `"filter"`, `"join"`, `"read_csv"`).               |
| `kwargs`    | `object` | No       | Keyword arguments to pass to the operation. Keys are parameter names, values can be literals or expression objects. |

**Valid operation examples:**

- `"read_csv"`, `"read_parquet"` - Data loading operations
- `"with_columns"`, `"select"` - Column operations
- `"filter"`, `"drop_nulls"` - Row filtering operations
- `"join"`, `"concat"` - Data combination operations
- `"group_by"`, `"agg"` - Aggregation operations

### Expression object

Represents a polars expression that can be nested and chained.

**Structure:**

```json
{
  "expr": "<string>",
  "on": <ExpressionObject | null>,
  "kwargs": {
    "<param_name>": <value>,
    ...
  }
}
```

**Fields:**

| Field    | Type               | Required | Description                                                                                                    |
| -------- | ------------------ | -------- | -------------------------------------------------------------------------------------------------------------- |
| `expr`   | `string`           | Yes      | Name of the polars expression function (e.g., `"col"`, `"lit"`, `"alias"`, `"str.to_datetime"`).               |
| `on`     | `ExpressionObject` | No       | Nested expression object that this expression operates on. If omitted, operates on the polars module directly. |
| `kwargs` | `object`           | No       | Keyword arguments specific to this expression function.                                                        |

**Common expression types:**

- `"col"` - Column reference (kwargs: `{"name": "column_name"}`)
- `"lit"` - Literal value (kwargs: `{"value": <any>}`)
- `"alias"` - Rename expression result (kwargs: `{"name": "new_name"}`)
- `"str.to_datetime"` - String to datetime conversion (kwargs: `{"format": "format_string"}`)
- `"cast"` - Type conversion (kwargs: `{"dtype": "data_type"}`)

### Path notation

Paths use dot notation to navigate the JSON structure:

- `"operations.0.kwargs.column_name"` - First operation's kwargs, column_name parameter
- `"operations.1.kwargs.expr_name.on.kwargs.value"` - Second operation's expression parameter's nested expression's kwargs value
- Array indices are zero-based numbers
- Object keys are referenced by name

### Example complete template

```json
{
  "name": "parse_date_column",
  "description": "Converts a string column to datetime format with configurable format string",
  "variables": [
    {
      "name": "Input File Path",
      "key": "input_file_path"
    },
    {
      "name": "Output File Path",
      "key": "output_file_path"
    }
  ],
  "inputs": [
    {
      "name": "Source column",
      "type": "column",
      "path": "operations.0.kwargs.parsed_date.on.on.kwargs.name",
      "default": "date_str"
    },
    {
      "name": "Date format",
      "type": "value",
      "path": "operations.0.kwargs.parsed_date.on.kwargs.format",
      "default": "%Y-%m-%d"
    },
    {
      "name": "Output column name",
      "type": "value",
      "path": "operations.0.kwargs.parsed_date.kwargs.name",
      "default": "parsed_date"
    }
  ],
  "outputs": [
    {
      "name": "Parsed date column",
      "type": "column",
      "path": "operations.0.kwargs.parsed_date.kwargs.name"
    }
  ],
  "operations": [
    {
      "operation": "with_columns",
      "kwargs": {
        "parsed_date": {
          "expr": "alias",
          "on": {
            "expr": "str.to_datetime",
            "on": {
              "expr": "col",
              "kwargs": {
                "name": "date_str"
              }
            },
            "kwargs": {
              "format": "%Y-%m-%d"
            }
          },
          "kwargs": {
            "name": "parsed_date"
          }
        }
      }
    }
  ]
}
```
