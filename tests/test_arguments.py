import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_as_config.config import run_config


def test_args_only():
    """Test using only positional arguments."""
    config = {
        "steps": [
            {
                "operation": "scan_csv", 
                "args": ["tests/test_data/xy.csv"]
            },
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


def test_kwargs_only():
    """Test using only keyword arguments (existing behavior)."""
    config = {
        "steps": [
            {
                "operation": "scan_csv", 
                "kwargs": {"source": "tests/test_data/xy.csv"}
            },
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


def test_args_and_kwargs_together():
    """Test using both positional and keyword arguments."""
    config = {
        "steps": [
            {
                "operation": "scan_csv",
                "args": ["tests/test_data/xy.csv"],
                "kwargs": {"has_header": True}
            },
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


def test_args_with_variables():
    """Test using variables in positional arguments."""
    config = {
        "variables": {
            "input_file": "tests/test_data/xy.csv"
        },
        "steps": [
            {
                "operation": "scan_csv",
                "args": ["$input_file"]
            },
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


def test_args_with_escaped_variables():
    """Test using escaped variables in positional arguments."""
    config = {
        "variables": {
            "input_file": "tests/test_data/xy.csv"
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/string_join.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "literal_text": {
                        "expr": "lit",
                        "args": ["$$input_file"],  # Should be literal "$input_file"
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
            "literal_text": ["$input_file", "$input_file", "$input_file"],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_args_with_expressions():
    """Test using expressions in positional arguments."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_plus_y": {
                        "expr": "add",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "args": [{"expr": "col", "kwargs": {"name": "y"}}],  # Using args instead of kwargs for "other"
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
            "x_plus_y": [3, 4, None, 8],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_mixed_args_types():
    """Test args with different data types."""
    config = {
        "variables": {
            "slice_offset": 1
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/string_join.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "sliced": {
                        "expr": "str.slice",
                        "on": {"expr": "col", "kwargs": {"name": "first"}},
                        "args": ["$slice_offset", 2],  # Mix of variable and literal
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


def test_empty_args_and_kwargs():
    """Test with empty args and kwargs."""
    config = {
        "steps": [
            {
                "operation": "scan_csv",
                "args": [],
                "kwargs": {"source": "tests/test_data/xy.csv"}
            },
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


def test_no_args_no_kwargs():
    """Test operation with neither args nor kwargs specified (should default to empty)."""
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {"operation": "collect"},  # No args or kwargs - should work
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
        }
    )
    assert_frame_equal(result, expected)  # Note: no .collect() here since it's already collected


def test_args_with_multiple_variables():
    """Test args with multiple variable substitutions."""
    config = {
        "variables": {
            "file_path": "tests/test_data/xy.csv",
            "has_header": True,
            "separator": ","
        },
        "steps": [
            {
                "operation": "read_csv",  # Using read_csv instead of scan_csv for this test
                "args": ["$file_path"],
                "kwargs": {
                    "has_header": "$has_header",
                    "separator": "$separator"
                }
            },
        ]
    }
    result = run_config(config)
    expected = pl.DataFrame(
        {
            "x": [1, 2, None, 4],
            "y": [2, 2, None, 4],
        }
    )
    assert_frame_equal(result, expected)


def test_complex_nested_args():
    """Test complex nested structure with args in expressions."""
    config = {
        "variables": {
            "multiplier": 3
        },
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "complex_calc": {
                        "expr": "add",
                        "on": {
                            "expr": "mul",
                            "on": {"expr": "col", "kwargs": {"name": "x"}},
                            "args": ["$multiplier"],  # Variable in nested args
                        },
                        "args": [1],  # Literal in args
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
            "complex_calc": [4, 7, None, 13],  # (x * 3) + 1
        }
    )
    assert_frame_equal(result.collect(), expected)
