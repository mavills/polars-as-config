import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_as_config.config import Config


class TestStringParsing:
    def test_parse_string(self):
        config = Config()
        assert config.parse_string("df", {}, None) == "df"

    def test_parse_variable(self):
        config = Config()
        assert config.parse_string("$a", {"a": "b"}, None) == "b"

    def test_parse_escaped_dollar_sign(self):
        config = Config()
        assert config.parse_string("$$a", {}, None) == "$a"

    def test_parse_escaped_dollar_sign_with_confusing_variable(self):
        config = Config()
        assert config.parse_string("$$a", {"a": "b", "$$a": "c"}, None) == "$a"

    def test_parse_polars_type_without_type_hint(self):
        config = Config()
        assert config.parse_string("Utf8", {}, None) == "Utf8"

    def test_parse_polars_type_with_type_hint(self):
        config = Config()
        assert config.parse_string("Utf8", {}, pl.DataType) == pl.Utf8

    def test_parse_polars_type_with_type_hint_and_variable(self):
        config = Config()
        assert config.parse_string("Utf8", {"Utf8": "b"}, pl.DataType) == pl.Utf8

    def test_parse_polars_dataframe_with_type_hint(self):
        """
        If it parses with a type hint, but without previously initializing the dataframe,
        we should raise an error.
        """
        config = Config()
        with pytest.raises(ValueError):
            config.parse_string("df", {}, pl.DataFrame)

    def test_parse_polars_dataframe_with_dataframe_but_no_type_hint(self):
        config = Config()
        config.current_dataframes = {"df": pl.DataFrame({"a": ["1", "2", "3"]})}
        assert config.parse_string("df", {}, None) == "df"

    def test_parse_polars_dataframe_with_dataframe_and_type_hint(self):
        config = Config()
        dataframe = pl.DataFrame({"a": ["1", "2", "3"]})
        config.current_dataframes = {"df": dataframe}
        result = config.parse_string("df", {}, pl.DataFrame)
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, dataframe)


class TestDictParsing:
    def test_parse_simple_dict(self):
        config = Config()
        result = config.parse_dict({"a": "b"}, {}, None)
        assert result == {"a": "b"}

    def test_parse_dict_with_expression(self):
        config = Config()
        result = config.parse_dict({"expr": "col", "kwargs": {"name": "a"}}, {}, None)
        assert isinstance(result, pl.Expr)
        assert str(result) == 'col("a")'

    def test_parse_dict_with_custom_function(self):
        def test_function():
            pass

        config = Config()
        config.add_custom_functions({"test_function": test_function})
        result = config.parse_dict({"custom_function": "test_function"}, {}, None)
        assert result == test_function

    def test_parse_dict_without_custom_function(self):
        config = Config()
        config.add_custom_functions({})
        with pytest.raises(ValueError):
            config.parse_dict({"custom_function": "test_function"}, {}, None)

    def test_parse_dict_with_nesting(self):
        config = Config()
        result = config.parse_dict(
            {
                "some_expression": {"expr": "col", "args": ["a"]},
                "some_list": [{"expr": "col", "args": ["b"]}],
                "some_nested_dict": {
                    "nested_string": "x",
                    "nested_list": [{"expr": "col", "args": ["y"]}],
                    "nested_expr": {"expr": "col", "args": ["z"]},
                },
            },
            {},
            None,
        )
        assert set(result.keys()) == {
            "some_expression",
            "some_list",
            "some_nested_dict",
        }
        assert str(result["some_expression"]) == 'col("a")'
        assert str(result["some_list"][0]) == 'col("b")'
        assert str(result["some_nested_dict"]["nested_list"][0]) == 'col("y")'
        assert str(result["some_nested_dict"]["nested_expr"]) == 'col("z")'
        assert result["some_nested_dict"]["nested_string"] == "x"


class TestListParsing:
    def test_parse_list_simple(self):
        config = Config()
        result = config.parse_list(["a", "b", "c"], {}, None)
        assert result == ["a", "b", "c"]

    def test_parse_list_with_nested_expression(self):
        config = Config()
        result = config.parse_list([{"expr": "col", "args": ["a"]}], {}, None)
        assert isinstance(result, list)
        assert len(result) == 1
        assert str(result[0]) == 'col("a")'

    def test_parse_list_with_type_hint_dtyes(self):
        config = Config()
        result = config.parse_list(["Utf8", "Int64", "Float64"], {}, pl.DataType)
        assert result == [pl.Utf8, pl.Int64, pl.Float64]

    def test_parse_list_with_list_type_hint_dtyes(self):
        config = Config()
        result = config.parse_list(["Utf8", "Int64", "Float64"], {}, list[pl.DataType])
        assert result == [pl.Utf8, pl.Int64, pl.Float64]

    def test_parse_list_without_type_hint_dtyes(self):
        config = Config()
        result = config.parse_list(["Utf8", "Int64", "Float64"], {}, None)
        assert result == ["Utf8", "Int64", "Float64"]

    def test_parse_list_with_type_hint_dataframe(self):
        # pl.concat
        config = Config()
        config.current_dataframes = {
            "df1": pl.DataFrame({"a": [1, 2, 3]}),
            "df2": pl.DataFrame({"a": [4, 5, 6]}),
        }
        result = config.parse_list(
            [{"expr": "concat", "args": [["df1", "df2"]]}],
            {},
            None,
        )
        expected = pl.concat(
            [config.current_dataframes["df1"], config.current_dataframes["df2"]]
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert_frame_equal(result[0], expected)
