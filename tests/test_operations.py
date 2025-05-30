from datetime import datetime, timedelta, timezone

import polars as pl
from polars.testing import assert_frame_equal

from polars_as_config.config import run_config


def test_eq():
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_eq_y": {
                        "expr": "eq",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"other": {"expr": "col", "kwargs": {"name": "y"}}},
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
            "x_eq_y": [False, True, None, True],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_add():
    config = {
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "with_columns",
                "kwargs": {
                    "x_add_int": {
                        "expr": "add",
                        "kwargs": {"other": 2},
                        "on": {
                            "expr": "col",
                            "kwargs": {"name": "x"},
                        },
                    },
                    "x_add_expr": {
                        "expr": "add",
                        "on": {"expr": "col", "kwargs": {"name": "x"}},
                        "kwargs": {"other": {"expr": "col", "kwargs": {"name": "y"}}},
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
            "x_add_int": [3, 4, None, 6],
            "x_add_expr": [3, 4, None, 8],
        }
    )
    assert_frame_equal(result.collect(), expected)


def test_join():
    config = {
        "steps": [
            {
                "operation": "scan_csv",
                "kwargs": {"source": "tests/test_data/string_join.csv"},
            },
            {
                "operation": "with_columns",
                "kwargs": {
                    "sliced": {
                        "expr": "str.slice",
                        "on": {"expr": "col", "kwargs": {"name": "first"}},
                        "kwargs": {
                            "offset": 1,
                            "length": 2,
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


def test_str_to_datetime():
    config = {
        "steps": [
            {
                "operation": "scan_csv",
                "kwargs": {"source": "tests/test_data/dates.csv"},
            },
            {
                "operation": "with_columns",
                "kwargs": {
                    "parsed_date": {
                        "expr": "str.to_datetime",
                        "on": {"expr": "col", "kwargs": {"name": "date_str"}},
                        "kwargs": {
                            "format": "%Y-%m-%d %H:%M%#z",
                        },
                    },
                },
            },
        ]
    }
    result = run_config(config)
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
