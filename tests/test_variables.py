import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_as_config.config import run_config


def test_variable_in_file_path():
    """Test using a variable for the CSV file path."""
    config = {
        "variables": {
            "input_file": "tests/test_data/xy.csv"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "$input_file"}},
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_variable_in_literal_value():
    """Test using variables for literal values in calculations."""
    config = {
        "variables": {
            "add_amount": 5,
            "multiply_factor": 2
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_plus_var": {
                        "expr": "add",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"other": "$add_amount"},
                    },
                    "x_times_var": {
                        "expr": "mul",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"other": "$multiply_factor"},
                    },
                },
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
            "x_plus_var": [6, 7, None, 9],
            "x_times_var": [2, 4, None, 8],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_variable_in_column_name():
    """Test using a variable for column names."""
    config = {
        "variables": {
            "source_column": "x",
            "target_column": "y"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "comparison": {
                        "expr": "eq",
                        "on": {"expr": "col", "kwargs": {"name": "$source_column"}},
                        "kwargs": {"other": {"expr": "col", "kwargs": {"name": "$target_column"}}},
                    }
                },
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
            "comparison": [False, True, None, True],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_variable_in_string_operations():
    """Test using variables in string operations and format strings."""
    config = {
        "variables": {
            "slice_offset": 1,
            "slice_length": 2,
            "date_format": "%Y-%m-%d %H:%M%#z"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/string_join.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "sliced": {
                        "expr": "str.slice",
                        "on": {"expr": "col", "kwargs": {"name": "first"}},
                        "kwargs": {
                            "offset": "$slice_offset",
                            "length": "$slice_length",
                        },
                    },
                },
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "first": ["hello", "good", "nice"],
            "second": ["world", "morning", "day"],
            "sliced": ["el", "oo", "ic"],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_variable_in_datetime_parsing():
    """Test using a variable for datetime format string."""
    config = {
        "variables": {
            "input_file_path": "tests/test_data/dates.csv",
            "date_format_string": "%Y-%m-%d %H:%M%#z",
            "source_col": "date_str"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "$input_file_path"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "parsed_date": {
                        "expr": "str.to_datetime",
                        "on": {"expr": "col", "kwargs": {"name": "$source_col"}},
                        "kwargs": {
                            "format": "$date_format_string",
                        },
                    },
                },
            },
        ]
    }
    result = run_config(config)
    # We expect the same result as the original datetime test
    from datetime import datetime, timedelta, timezone
    expected = pl.DataFrame(
        {
            "date_str": [
                "2023-01-01 01:00Z",
                "2023-01-01 02:00Z",
                "2023-01-01 03:00+0100",
                "2023-01-01 04:00-0500",
            ],
            "parsed_date": [
                datetime(2023, 1, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 2, 0, tzinfo=timezone.utc),
                datetime(2023, 1, 1, 3, 0, tzinfo=timezone(timedelta(hours=1))),
                datetime(2023, 1, 1, 4, 0, tzinfo=timezone(timedelta(hours=-5))),
            ],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_multiple_variables_same_value():
    """Test using multiple variables that reference the same value."""
    config = {
        "variables": {
            "common_addend": 3,
            "also_three": 3
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_plus_first": {
                        "expr": "add",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"other": "$common_addend"},
                    },
                    "x_plus_second": {
                        "expr": "add",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"other": "$also_three"},
                    },
                },
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
            "x_plus_first": [4, 5, None, 7],
            "x_plus_second": [4, 5, None, 7],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_escaped_dollar_sign():
    """Test that escaped dollar signs ($$) are handled correctly."""
    config = {
        "variables": {
            "my_var": "should_not_be_used"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/string_join.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "literal_dollar": {
                        "expr": "lit",
                        "kwargs": {"value": "$$my_var"},  # Should result in "$my_var", not variable substitution
                    },
                },
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "first": ["hello", "good", "nice"],
            "second": ["world", "morning", "day"],
            "literal_dollar": ["$my_var", "$my_var", "$my_var"],  # Should be literal "$my_var"
        }
    )
    # This should now pass with the escaping mechanism implemented
    assert_frame_equal(result.collect(), expected)


def test_mixed_escaped_and_unescaped_variables():
    """Test mixing escaped dollar signs with actual variable substitution."""
    config = {
        "variables": {
            "actual_value": 42.0,
            "file_path": "tests/test_data/xy.csv"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "$file_path"}},  # Real variable
            {
                "operation": "with_columns",
                "kwargs": {
                    "escaped_text": {
                        "expr": "lit", 
                        "kwargs": {"value": "$$actual_value"}  # Escaped - should be "$actual_value"
                    },
                    "real_value": {
                        "expr": "lit",
                        "kwargs": {"value": "$actual_value"}  # Real variable - should be 42
                    },
                },
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
            "escaped_text": ["$actual_value", "$actual_value", "$actual_value", "$actual_value"],
            "real_value": [42.0, 42.0, 42.0, 42.0],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_missing_variable_error():
    """Test that referencing a non-existent variable raises an appropriate error."""
    config = {
        "variables": {
            "existing_var": "tests/test_data/xy.csv"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "$nonexistent_var"}},
        ]
    }
    with pytest.raises(KeyError):
        run_config(config)


def test_no_variables_section():
    """Test that configs work fine without a variables section."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
        }
    )
    assert_frame_equal(result.collect(), expected)
