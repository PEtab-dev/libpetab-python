import warnings

import pytest


def test_deprecated_global():
    with pytest.warns(DeprecationWarning):
        from petab import Problem

    with pytest.warns(DeprecationWarning):  # noqa PT031
        import petab

        petab.Problem()

    with pytest.warns(DeprecationWarning):  # noqa PT031
        import petab.parameters

        petab.parameters  # noqa

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from petab.v1 import Problem

        Problem()

        import petab.v1.parameters
