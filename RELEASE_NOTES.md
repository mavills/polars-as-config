# Release Notes

## Version 1.2.0 - Multiple Dataframes Support

### 🎉 New Features

#### Enhanced Dataframe Management with `dataframe_in` and `dataframe_out`

Version 1.2.0 introduces a more explicit and powerful way to manage multiple
dataframes in your configurations. Previously, the `dataframe` field implicitly
served as both the input and output dataframe identifier. Now, you can
explicitly control which dataframe a step reads from and which dataframe it
writes to.

**New Syntax:**

```json
{
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe_out": "customers",
      "kwargs": { "source": "customers.csv" }
    },
    {
      "operation": "scan_csv",
      "dataframe_out": "orders",
      "kwargs": { "source": "orders.csv" }
    },
    {
      "operation": "join",
      "dataframe_in": "customers",
      "dataframe_out": "customer_orders",
      "kwargs": {
        "other": "orders",
        "on": "customer_id"
      }
    }
  ]
}
```

**Key Benefits:**

1. **Explicit Control**: Clearly specify which dataframe each operation reads
   from (`dataframe_in`) and writes to (`dataframe_out`)
2. **Immutability**: Create new dataframes without modifying existing ones by
   specifying different input and output dataframes
3. **Clarity**: Makes the data flow in complex multi-dataframe pipelines more
   readable and maintainable
4. **Flexibility**: Enables advanced patterns like creating multiple derived
   dataframes from a single source

**Behavior:**

- `dataframe_in`: Specifies which dataframe the operation should read from. If
  omitted, defaults to `None` (the default dataframe).
- `dataframe_out`: Specifies which dataframe the operation result should be
  assigned to. If omitted, defaults to `None` (the default dataframe).
- If neither is specified, both default to `None`, maintaining backward
  compatibility with single-dataframe configurations.

### 📚 Improved Documentation

- New dedicated **Dataframes.md** wiki page covering all aspects of dataframe
  management
- Comprehensive examples showing single and multiple dataframe usage patterns
- Clear migration guide from legacy syntax to new syntax

### 🔄 Legacy Support

#### The `dataframe` Field (Legacy)

For backward compatibility, the original `dataframe` field continues to work.
When used, it sets both `dataframe_in` and `dataframe_out` to the same value:

```json
{
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe": "customers",
      "kwargs": { "source": "customers.csv" }
    },
    {
      "operation": "with_columns",
      "dataframe": "customers",
      "kwargs": {
        "customer_type": { "expr": "lit", "kwargs": { "value": "premium" } }
      }
    }
  ]
}
```

This is equivalent to:

```json
{
  "steps": [
    {
      "operation": "scan_csv",
      "dataframe_out": "customers",
      "kwargs": { "source": "customers.csv" }
    },
    {
      "operation": "with_columns",
      "dataframe_in": "customers",
      "dataframe_out": "customers",
      "kwargs": {
        "customer_type": { "expr": "lit", "kwargs": { "value": "premium" } }
      }
    }
  ]
}
```

**Important:** You cannot mix the legacy `dataframe` syntax with the new
`dataframe_in`/`dataframe_out` syntax in the same step. Doing so will raise a
`ValueError`.

### ⚠️ Breaking Changes

None. This release is fully backward compatible. Existing configurations using
the `dataframe` field will continue to work without modification.

### 🔧 Technical Details

- Automatic dataframe reference resolution now supports both legacy and new
  syntax
- Type-hint based detection ensures string references to dataframes are
  correctly resolved in operation parameters
- Clear error messages when referencing non-existent dataframes
- Return value from `Config.run_config()` is now a dictionary mapping dataframe
  names to their final states

### 📖 Migration Guide

**From Legacy Syntax:**

```json
{
  "operation": "with_columns",
  "dataframe": "my_data",
  "kwargs": {...}
}
```

**To New Syntax (when modifying in-place):**

```json
{
  "operation": "with_columns",
  "dataframe_in": "my_data",
  "dataframe_out": "my_data",
  "kwargs": {...}
}
```

**To New Syntax (when creating a new dataframe):**

```json
{
  "operation": "with_columns",
  "dataframe_in": "my_data",
  "dataframe_out": "my_data_transformed",
  "kwargs": {...}
}
```
