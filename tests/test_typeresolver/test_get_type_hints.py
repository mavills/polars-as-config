import polars as pl

from polars_as_config.config import Config


def test_get_parameter_types():
    def test_function(a: str, b: int, default: str = "default"):
        pass

    config = Config()
    paramters = config._get_parameter_types(test_function)
    assert isinstance(paramters["a"].annotation, type(str))
    assert isinstance(paramters["b"].annotation, type(int))
    assert isinstance(paramters["default"].annotation, type(str))


def test_get_parameter_types_with_expr():
    """
    Polars has a special case where the type hint is a Expr object.
    Normal type resolving doesn't work for this.

    This is for example a problem when using the pl.col object.
    """

    def test_function(a: pl.Expr):
        pass

    config = Config()
    paramters = config._get_parameter_types(test_function)
    assert isinstance(paramters["a"].annotation, type(pl.Expr))


def test_get_parameter_types_of_col():
    """
    Polars has a special case where the type hint is a Col object.
    Normal type resolving doesn't work for this.

    For Col, this is hard-coded to be a string.
    """
    config = Config()
    paramters = config._get_parameter_types(pl.col)
    assert isinstance(paramters["name"].annotation, str)


def test_get_parameter_types_with_lit():
    def test_function(a: pl.lit):
        pass

    config = Config()
    paramters = config._get_parameter_types(test_function)
    assert isinstance(paramters["a"].annotation, type(pl.lit))


def test_get_parameter_types_of_lit():
    config = Config()
    paramters = config._get_parameter_types(pl.lit)
    # check for string, but is typed as Any. For lit we are generally safe.
    assert isinstance(paramters["value"].annotation, str)
