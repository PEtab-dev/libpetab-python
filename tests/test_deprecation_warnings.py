import warnings

import pytest


def test_deprecated_global():
    with pytest.warns(DeprecationWarning):
        from petab import Problem  # noqa

    with pytest.warns(DeprecationWarning):
        import petab

        petab.Problem()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from petab.v1 import Problem  # noqa

        Problem()
