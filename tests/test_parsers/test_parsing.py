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
    pass


class TestListParsing:
    pass
