import inspect
from typing import ForwardRef, Union

import polars as pl
from polars._typing import PolarsType
from typing_extensions import TypeVar

from polars_as_config.config import Config


class TestIsType:
    def test_is_type_none(self):
        """
        None is not a type.
        """
        config = Config()
        assert not config._is_type(None, str)

    def test_is_type_str(self):
        """
        A basic type, str.
        """
        config = Config()
        config._is_type(str, str)

    def test_is_not_type_str(self):
        """
        A basic type, str.
        """
        config = Config()
        assert not config._is_type(int, str)

    def test_union_type(self):
        """
        A union type must check all the types in the union.
        """
        config = Config()
        config._is_type(Union[str, int], str)

    def test_is_not_union_type(self):
        """
        A union type must check all the types in the union.
        """
        config = Config()
        assert not config._is_type(int, Union[str, int])

    def test_is_type_dataframe(self):
        """
        We specifically need to check for dataframes.
        """
        config = Config()
        assert config._is_type(pl.DataFrame, pl.DataFrame)

    def test_is_not_type_dataframe(self):
        """
        We specifically need to check for dataframes.
        """
        config = Config()
        assert not config._is_type(pl.Series, pl.DataFrame)

    def test_is_type_datatype(self):
        """
        We specifically need to check for datatypes, polars ones.
        """
        config = Config()
        assert config._is_type(pl.DataType, pl.DataType)

    def test_is_not_type_datatype(self):
        """
        We specifically need to check for datatypes, polars ones.
        """
        config = Config()
        assert not config._is_type(pl.DataType, pl.Series)

    # def test_forward_ref(self):
    #     """
    #     We specifically need to check for forward references.
    #     These occur when we want to prevent circular imports.
    #     In polars, this happens on DataType.
    #     """
    #     config = Config()
    #     assert config._is_type(ForwardRef("DataType"), pl.DataType)

    def test_forward_ref_union(self):
        """
        Forward references can also be in unions.
        """
        config = Config()
        assert config._is_type(Union[ForwardRef("str"), int], str)

    def test_type_vars(self):
        """
        A specific case is pl.concat for example, which has a typing variable.
        """
        config = Config()
        assert config._is_type(
            PolarsType,
            pl.DataFrame,
        )
