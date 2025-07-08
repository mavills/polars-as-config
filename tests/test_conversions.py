from polars_as_config.helpers import json_to_polars


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
    code = json_to_polars(config["steps"], format="dataframe")
    expected = """df = polars.read_csv('data.csv')
df = df.with_columns(polars.add(polars.col('a'), 10).alias('new_column', brrr='a'))
df = df.collect()"""
    print(code)
    assert code == expected
