import polars as pl
from polars.testing import assert_frame_equal

from polars_as_config.config import Config


def change_to_constant(x: str) -> str:
    return "constant"


def test_attribute_as_argument():
    # .map_elements(create_hashed_ids, return_dtype=str)
    config = {
        "variables": {"multiplier": 3},
        "steps": [
            {"operation": "scan_csv", "kwargs": {"source": "tests/test_data/xy.csv"}},
            {
                "operation": "select",
                "args": [
                    {
                        "expr": "map_elements",
                        "on": {
                            "expr": "col",
                            "kwargs": {"name": "x"},
                        },
                        "args": [{"custom_function": "change_to_constant"}],
                        "kwargs": {"return_dtype": "Utf8"},
                    },
                ],
            },
        ],
    }
    df = pl.scan_csv("tests/test_data/xy.csv")
    df = df.select(x=pl.col("x").map_elements(change_to_constant, return_dtype=pl.Utf8))

    expected = pl.DataFrame({"x": ["constant", "constant", None, "constant"]})

    assert_frame_equal(df.collect(), expected)

    result = (
        Config()
        .add_custom_functions({"change_to_constant": change_to_constant})
        .run_config(config)
    )
    assert_frame_equal(result[None].collect(), expected)
