from polars_as_config.polars_to_json import PolarsToJson


def test_single_operation():
    code = "df = polars.read_csv()"
    expected = [
        {
            "operation": "read_csv",
            "args": [],
            "kwargs": {},
            "dataframe_out": "df",
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
            "dataframe_out": "df",
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
            "dataframe_out": "df",
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
            "dataframe_out": "df",
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
            "dataframe_out": "df",
        },
        {
            "operation": "sum",
            "args": [],
            "kwargs": {},
            "dataframe_out": "df",
            "dataframe_in": "df",
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
            "dataframe_out": "df1",
        },
        {
            "operation": "read_csv",
            "args": ["data2.csv"],
            "kwargs": {},
            "dataframe_out": "df2",
        },
        {
            "operation": "sum",
            "args": [],
            "kwargs": {},
            "dataframe_out": "df1",
            "dataframe_in": "df1",
        },
        {
            "operation": "sum",
            "args": [],
            "kwargs": {},
            "dataframe_out": "df2",
            "dataframe_in": "df2",
        },
        {
            "operation": "join",
            "args": ["df1"],
            "kwargs": {},
            "dataframe_out": "df2",
            "dataframe_in": "df2",
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
            "dataframe_out": "df",
            "dataframe_in": "df",
        },
    ]
    p2j = PolarsToJson()
    p2j.dataframes.add("df")
    assert p2j.polars_to_json(code) == expected


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
            "dataframe_out": "df",
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
            "dataframe_out": "df",
            "dataframe_in": "df",
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
            "dataframe_out": "df",
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
            "dataframe_out": "df",
            "dataframe_in": "df",
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
            "dataframe_out": "df",
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
            "dataframe_out": "df",
            "dataframe_in": "df",
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
            "dataframe_out": "df",
            "dataframe_in": "df",
        }
    ]
    p2j = PolarsToJson()
    p2j.dataframes.add("df")
    assert p2j.polars_to_json(code) == expected


def test_arguments_type_list():
    code = "df = df.with_columns(y_contains_o=pl.col('y').str.contains(['o', 'a']))"
    expected = [
        {
            "operation": "with_columns",
            "args": [],
            "kwargs": {
                "y_contains_o": {
                    "expr": "str.contains",
                    "args": [["o", "a"]],
                    "kwargs": {},
                    "on": {"expr": "col", "args": ["y"], "kwargs": {}},
                }
            },
            "dataframe_out": "df",
            "dataframe_in": "df",
        }
    ]
    p2j = PolarsToJson()
    p2j.dataframes.add("df")
    assert p2j.polars_to_json(code) == expected


def test_arguments_type_dict():
    code = "df = df.with_columns(y_contains_o=pl.col('y').str.contains({'o': 'a'}))"
    expected = [
        {
            "operation": "with_columns",
            "args": [],
            "kwargs": {
                "y_contains_o": {
                    "expr": "str.contains",
                    "args": [{"o": "a"}],
                    "kwargs": {},
                    "on": {"expr": "col", "args": ["y"], "kwargs": {}},
                }
            },
            "dataframe_out": "df",
            "dataframe_in": "df",
        }
    ]
    p2j = PolarsToJson()
    p2j.dataframes.add("df")
    assert p2j.polars_to_json(code) == expected


def test_attribute_as_argument():
    code = """df = df.select(
        x=pl.col("x").map_elements(pl.col("y"), return_dtype=pl.Utf8)
    )"""
    expected = [
        {
            "operation": "select",
            "args": [],
            "kwargs": {
                "x": {
                    "args": [{"args": ["y"], "kwargs": {}, "expr": "col"}],
                    "kwargs": {"return_dtype": "Utf8"},
                    "on": {"args": ["x"], "kwargs": {}, "expr": "col"},
                    "expr": "map_elements",
                }
            },
            "dataframe_out": "df",
            "dataframe_in": "df",
        }
    ]
    p2j = PolarsToJson()
    p2j.dataframes.add("df")
    assert p2j.polars_to_json(code) == expected
