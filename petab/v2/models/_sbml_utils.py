"""Private utility functions for SBML handling."""

import libsbml

retval_to_str = {
    getattr(libsbml, attr): attr
    for attr in (
        "LIBSBML_DUPLICATE_OBJECT_ID",
        "LIBSBML_INDEX_EXCEEDS_SIZE",
        "LIBSBML_INVALID_ATTRIBUTE_VALUE",
        "LIBSBML_INVALID_OBJECT",
        "LIBSBML_INVALID_XML_OPERATION",
        "LIBSBML_LEVEL_MISMATCH",
        "LIBSBML_NAMESPACES_MISMATCH",
        "LIBSBML_OPERATION_FAILED",
        "LIBSBML_UNEXPECTED_ATTRIBUTE",
        "LIBSBML_PKG_UNKNOWN",
        "LIBSBML_PKG_VERSION_MISMATCH",
        "LIBSBML_PKG_CONFLICTED_VERSION",
    )
}


def check(res: int):
    """Check the return value of a libsbml function that returns a status code.

    :param res: The return value to check.
    :raises RuntimeError: If the return value indicates an error.
    """
    if res != libsbml.LIBSBML_OPERATION_SUCCESS:
        raise RuntimeError(f"libsbml error: {retval_to_str.get(res, res)}")


def add_sbml_parameter(
    model: libsbml.Model,
    id_: str,
    value: float = None,
    constant: bool = None,
) -> libsbml.Parameter:
    """Add a parameter to the SBML model."""
    param = model.createParameter()

    check(param.setId(id_))

    if value is not None:
        check(param.setValue(value))

    if constant is not None:
        check(param.setConstant(constant))

    return param
