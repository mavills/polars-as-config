import hashlib
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_as_config.config import Config, run_config


def hash_row(row: dict) -> str:
    """
    Concatenate all values in a row and create a SHA-256 hash.
    """
    # Convert all values to strings and concatenate
    row_str = "".join(str(val) for val in row.values())
    # Create SHA-256 hash
    return hashlib.sha256(row_str.encode()).hexdigest()


def multiply_by_two(value: int) -> int:
    """Simple multiplication function for testing."""
    return value * 2


def format_name(row: dict) -> str:
    """Combine first and last name with proper formatting."""
    return f"{row['first'].title()} {row['second'].title()}"


def calculate_age_category(age: int) -> str:
    """Categorize age into groups."""
    if age < 18:
        return "minor"
    elif age < 65:
        return "adult"
    else:
        return "senior"


def test_hash_function_basic():
    """Test the hash function with basic data - main focus from user example."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "row_hash": {
                        "expr": "map_elements",
                        "on": {"expr": "struct", "args": [{"expr": "all"}]},
                        "kwargs": {"function": {"custom_function": "hash_row"}},
                    }
                },
            },
        ]
    }

    # Create config with custom function
    custom_config = Config().add_custom_functions({"hash_row": hash_row})
    result = custom_config.run_config(config)

    # Verify that hash column was added and contains valid hashes
    collected = result.collect()
    assert "row_hash" in collected.columns
    assert len(collected["row_hash"]) == 4

    # Check that all hash values are 64-character hex strings (SHA-256)
    for hash_val in collected["row_hash"]:
        if hash_val is not None:  # Skip None values if any
            assert len(hash_val) == 64
            assert all(c in "0123456789abcdef" for c in hash_val)


def test_hash_function_with_transformations():
    """Test hash function with data transformations like in user example."""
    config = {
        "steps": [
            {
                "operation": "scan_csv",
                "kwargs": {"source": "tests/test_data/string_join.csv"},
            },
            {
                "operation": "with_columns",
                "kwargs": {
                    "source": {"expr": "lit", "args": ["workday-resume-document"]},
                },
            },
            {
                "operation": "rename",
                "kwargs": {"mapping": {"first": "name", "second": "category"}},
            },
            {
                "operation": "with_columns",
                "kwargs": {
                    "row_hash": {
                        "expr": "map_elements",
                        "on": {"expr": "struct", "args": [{"expr": "all"}]},
                        "kwargs": {"function": {"custom_function": "hash_row"}},
                    }
                },
            },
        ]
    }

    custom_config = Config().add_custom_functions({"hash_row": hash_row})
    result = custom_config.run_config(config)
    collected = result.collect()

    expected_columns = ["name", "category", "source", "row_hash"]
    assert set(collected.columns) == set(expected_columns)
    assert all(len(hash_val) == 64 for hash_val in collected["row_hash"])


def test_simple_custom_function():
    """Test a simple custom function that multiplies values."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_doubled": {
                        "expr": "map_elements",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"function": {"custom_function": "multiply_by_two"}},
                    }
                },
            },
        ]
    }

    custom_config = Config().add_custom_functions({"multiply_by_two": multiply_by_two})
    result = custom_config.run_config(config)

    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
            "x_doubled": [2, 4, None, 8],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_custom_function_with_struct():
    """Test custom function that operates on row data (struct)."""
    config = {
        "steps": [
            {
                "operation": "scan_csv",
                "kwargs": {"source": "tests/test_data/string_join.csv"},
            },
            {
                "operation": "with_columns",
                "kwargs": {
                    "full_name": {
                        "expr": "map_elements",
                        "on": {
                            "expr": "struct",
                            "args": [
                                {"expr": "col", "kwargs": {"name": "first"}},
                                {"expr": "col", "kwargs": {"name": "second"}},
                            ],
                        },
                        "kwargs": {"function": {"custom_function": "format_name"}},
                    }
                },
            },
        ]
    }

    custom_config = Config().add_custom_functions({"format_name": format_name})
    result = custom_config.run_config(config)

    expected = pl.DataFrame(
        {
            "first": ["hello", "good", "nice"],
            "second": ["world", "morning", "day"],
            "full_name": ["Hello World", "Good Morning", "Nice Day"],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_multiple_custom_functions():
    """Test using multiple custom functions in the same config."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_doubled": {
                        "expr": "map_elements",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"function": {"custom_function": "multiply_by_two"}},
                    },
                    "row_hash": {
                        "expr": "map_elements",
                        "on": {"expr": "struct", "args": [{"expr": "all"}]},
                        "kwargs": {"function": {"custom_function": "hash_row"}},
                    },
                },
            },
        ]
    }

    custom_config = Config().add_custom_functions(
        {"multiply_by_two": multiply_by_two, "hash_row": hash_row}
    )
    result = custom_config.run_config(config)
    collected = result.collect()

    # Check both custom functions worked
    assert "x_doubled" in collected.columns
    assert "row_hash" in collected.columns
    assert collected["x_doubled"][0] == 2  # 1 * 2
    assert len(collected["row_hash"][0]) == 64  # Valid hash


def test_custom_function_with_variables():
    """Test custom functions combined with variable substitution. This should not work; this is not implemented that way."""
    config = {
        "variables": {"multiplier_func": "multiply_by_two"},
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_doubled": {
                        "expr": "map_elements",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"function": {"custom_function": "$multiplier_func"}},
                    }
                },
            },
        ],
    }

    custom_config = Config().add_custom_functions({"multiply_by_two": multiply_by_two})
    with pytest.raises(KeyError, match="'\\$multiplier_func'"):
        custom_config.run_config(config)


def test_custom_function_error_handling():
    """Test error handling when custom function is not found."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "result": {
                        "expr": "map_elements",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {
                            "function": {"custom_function": "nonexistent_function"}
                        },
                    }
                },
            },
        ]
    }

    custom_config = Config()  # No custom functions added
    with pytest.raises(KeyError):
        custom_config.run_config(config)


def test_hash_function_deterministic():
    """Test that hash function produces consistent results for same input."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "row_hash": {
                        "expr": "map_elements",
                        "on": {"expr": "struct", "args": [{"expr": "all"}]},
                        "kwargs": {"function": {"custom_function": "hash_row"}},
                    }
                },
            },
        ]
    }

    custom_config = Config().add_custom_functions({"hash_row": hash_row})

    # Run the same config twice
    result1 = custom_config.run_config(config).collect()
    result2 = custom_config.run_config(config).collect()

    # Hashes should be identical
    assert_frame_equal(result1, result2)


def test_config_builder_pattern():
    """Test that the builder pattern works for adding custom functions."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_doubled": {
                        "expr": "map_elements",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"function": {"custom_function": "multiply_by_two"}},
                    }
                },
            },
        ]
    }

    # Test builder pattern
    result = (
        Config()
        .add_custom_functions({"multiply_by_two": multiply_by_two})
        .run_config(config)
    )

    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
            "x_doubled": [2, 4, None, 8],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_hash_function_with_different_data_types():
    """Test hash function with mixed data types to ensure proper string conversion."""
    # Create a test DataFrame with mixed types
    test_data = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "float_col": [1.1, 2.2, 3.3],
        }
    ).lazy()

    config = {
        "steps": [
            {
                "operation": "with_columns",
                "kwargs": {
                    "row_hash": {
                        "expr": "map_elements",
                        "on": {"expr": "struct", "args": [{"expr": "all"}]},
                        "kwargs": {"function": {"custom_function": "hash_row"}},
                    }
                },
            }
        ]
    }

    custom_config = Config().add_custom_functions({"hash_row": hash_row})

    # Manually test the function works with the initial data
    result = custom_config.handle_step(test_data, config["steps"][0], {})
    collected = result.collect()

    assert "row_hash" in collected.columns
    assert len(collected["row_hash"]) == 3
    # Each hash should be unique since the rows are different
    assert len(set(collected["row_hash"])) == 3
