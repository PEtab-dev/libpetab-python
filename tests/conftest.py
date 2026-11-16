import pytest


def _cleanup_pysb_exports():
    """Remove all symbols exported by :class:`pysb.SelfExporter`.

    :meth:`pysb.SelfExporter.cleanup` does not remove a model that was
    exported under an explicit name, so remove that one first.
    Fails if any other exported PySB symbols remain in the target namespace.
    """
    import pysb

    exporter = pysb.SelfExporter
    namespace = exporter.target_globals
    module = exporter.target_module
    model = exporter.default_model
    if model is not None and namespace.get(model.name) is model:
        del namespace[model.name]
    exporter.cleanup()

    if namespace is None:
        return

    # objects bound under their own name were exported by `SelfExporter`
    leaked = [
        name
        for name, value in namespace.items()
        if isinstance(value, pysb.Model | pysb.Component)
        and value.name == name
    ]
    for name in leaked:
        del namespace[name]
    assert not leaked, f"Leaked PySB symbols in {module}: {leaked}"


@pytest.fixture
def clean_pysb_exports():
    """Clean up PySB auto-exported symbols before and after the test.

    Yields the cleanup function for use within the test, e.g., before
    defining another model.
    """
    _cleanup_pysb_exports()
    yield _cleanup_pysb_exports
    _cleanup_pysb_exports()
