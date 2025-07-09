from polars_as_config.polars_to_json import PolarsToJson


def test_single_operation():
    code = "df = polars.read_csv()"
    expected = [
        {
            "operation": "read_csv",
            "args": [],
            "kwargs": {},
            "dataframe": "df",
        }
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_single_operation_with_constant_args():
    code = 'df = polars.read_csv("data.csv")'
    expected = [
        {
            "operation": "read_csv",
            "args": ["data.csv"],
            "kwargs": {},
            "dataframe": "df",
        }
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_single_operation_with_constant_kwargs():
    code = 'df = polars.read_csv(source="data.csv")'
    expected = [
        {
            "operation": "read_csv",
            "args": [],
            "kwargs": {"source": "data.csv"},
            "dataframe": "df",
        }
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_single_operation_with_constant_args_and_kwargs():
    code = 'df = polars.read_csv("data.csv", has_header=True)'
    expected = [
        {
            "operation": "read_csv",
            "args": ["data.csv"],
            "kwargs": {"has_header": True},
            "dataframe": "df",
        }
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_multiple_operations():
    code = """df = polars.read_csv("data.csv")
df = df.sum()"""
    expected = [
        {
            "operation": "read_csv",
            "args": ["data.csv"],
            "kwargs": {},
            "dataframe": "df",
        },
        {
            "operation": "sum",
            "args": [],
            "kwargs": {},
            "dataframe": "df",
        },
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_multiple_operations_with_multiple_dataframes():
    code = """df1 = polars.read_csv("data.csv")
df2 = polars.read_csv("data2.csv")
df1 = df1.sum()
df2 = df2.sum()
df2 = df2.join(df1)"""
    expected = [
        {
            "operation": "read_csv",
            "args": ["data.csv"],
            "kwargs": {},
            "dataframe": "df1",
        },
        {
            "operation": "read_csv",
            "args": ["data2.csv"],
            "kwargs": {},
            "dataframe": "df2",
        },
        {
            "operation": "sum",
            "args": [],
            "kwargs": {},
            "dataframe": "df1",
        },
        {
            "operation": "sum",
            "args": [],
            "kwargs": {},
            "dataframe": "df2",
        },
        {
            "operation": "join",
            "args": ["df1"],
            "kwargs": {},
            "dataframe": "df2",
        },
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_single_nested_operation():
    code = "df = df.select(pl.col('a').add(1).alias('b'))"
    expected = [
        {
            "operation": "select",
            "args": [
                {
                    "expr": "alias",
                    "args": [
                        "b",
                    ],
                    "kwargs": {},
                    "on": {
                        "expr": "add",
                        "args": [
                            1,
                        ],
                        "kwargs": {},
                        "on": {"expr": "col", "args": ["a"], "kwargs": {}},
                    },
                }
            ],
            "kwargs": {},
            "dataframe": "df",
        },
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_multiple_nested_operations():
    code = """df = pl.select(pl.col('a').add(1).alias('b'))
df = df.select(pl.col('b').add(1).alias('c'))"""
    expected = [
        {
            "operation": "select",
            "args": [
                {
                    "expr": "alias",
                    "args": [
                        "b",
                    ],
                    "kwargs": {},
                    "on": {
                        "expr": "add",
                        "args": [
                            1,
                        ],
                        "kwargs": {},
                        "on": {"expr": "col", "args": ["a"], "kwargs": {}},
                    },
                }
            ],
            "kwargs": {},
            "dataframe": "df",
        },
        {
            "operation": "select",
            "args": [
                {
                    "expr": "alias",
                    "args": [
                        "c",
                    ],
                    "kwargs": {},
                    "on": {
                        "expr": "add",
                        "args": [
                            1,
                        ],
                        "kwargs": {},
                        "on": {"expr": "col", "args": ["b"], "kwargs": {}},
                    },
                }
            ],
            "kwargs": {},
            "dataframe": "df",
        },
    ]
    assert PolarsToJson().polars_to_json(code) == expected


def test_custom_function_with_custom_functions():
    code = """df = pl.read_csv(source="tests/test_data/xy.csv")
df = df.with_columns(row_hash=pl.struct(pl.all()).map_elements(function=hash_row))"""
    expected = [
        {
            "operation": "read_csv",
            "args": [],
            "kwargs": {"source": "tests/test_data/xy.csv"},
            "dataframe": "df",
        },
        {
            "operation": "with_columns",
            "args": [],
            "kwargs": {
                "row_hash": {
                    "expr": "map_elements",
                    "on": {
                        "expr": "struct",
                        "args": [{"expr": "all", "kwargs": {}, "args": []}],
                        "kwargs": {},
                    },
                    "args": [],
                    "kwargs": {"function": {"custom_function": "hash_row"}},
                }
            },
            "dataframe": "df",
        },
    ]
    assert PolarsToJson(custom_functions={"hash_row"}).polars_to_json(code) == expected


def test_custom_function_with_allow_function_discovery():
    code = """df = pl.read_csv(source="tests/test_data/xy.csv")
df = df.with_columns(row_hash=pl.struct(pl.all()).map_elements(function=hash_row))"""
    expected = [
        {
            "operation": "read_csv",
            "args": [],
            "kwargs": {"source": "tests/test_data/xy.csv"},
            "dataframe": "df",
        },
        {
            "operation": "with_columns",
            "args": [],
            "kwargs": {
                "row_hash": {
                    "expr": "map_elements",
                    "on": {
                        "expr": "struct",
                        "args": [{"expr": "all", "kwargs": {}, "args": []}],
                        "kwargs": {},
                    },
                    "args": [],
                    "kwargs": {"function": {"custom_function": "hash_row"}},
                }
            },
            "dataframe": "df",
        },
    ]
    assert PolarsToJson(allow_function_discovery=True).polars_to_json(code) == expected


def test_polars_string_expression():
    code = "df = df.with_columns(y_contains_o=pl.col('y').str.contains('o'))"
    expected = [
        {
            "operation": "with_columns",
            "args": [],
            "kwargs": {
                "y_contains_o": {
                    "expr": "str.contains",
                    "args": ["o"],
                    "kwargs": {},
                    "on": {"expr": "col", "args": ["y"], "kwargs": {}},
                }
            },
            "dataframe": "df",
        }
    ]
    assert PolarsToJson().polars_to_json(code) == expected
