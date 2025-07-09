from polars_as_config.json_to_polars import JsonToPolars
from polars_as_config.polars_to_json import PolarsToJson


def test_to_code():
    config = {
        "steps": [
            {"operation": "read_csv", "args": ["data.csv"]},
            {
                "operation": "with_columns",
                "args": [
                    {
                        "expr": "alias",
                        "on": {
                            "expr": "add",
                            "args": [{"expr": "col", "args": ["a"]}, 10],
                        },
                        "args": ["new_column"],
                        "kwargs": {"brrr": "a"},
                    }
                ],
            },
            {"operation": "collect"},
        ]
    }
    code = JsonToPolars().json_to_polars(config["steps"], format="dataframe")
    expected = """df = polars.read_csv('data.csv')
df = df.with_columns(polars.add(polars.col('a'), 10).alias('new_column', brrr='a'))
df = df.collect()"""
    print(code)
    assert code == expected


def test_polars_to_json_to_polars():
    expected = """df = polars.read_csv('data.csv')
df = df.with_columns(polars.add(polars.col('a'), 10).alias('new_column', brrr='a'))
df = df.collect()"""
    code = JsonToPolars().json_to_polars(
        PolarsToJson().polars_to_json(expected), format="dataframe"
    )
    assert code == expected
