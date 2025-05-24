"""Basic tests for the package."""

from polars_as_config import __version__


def test_version():
    """Test that version is a string."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0 