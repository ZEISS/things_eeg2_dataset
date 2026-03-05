import pytest


def test_deprecation_warnings() -> None:
    with pytest.warns(DeprecationWarning):
        pass
