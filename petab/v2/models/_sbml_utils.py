"""Private utility functions for SBML handling."""

import libsbml
import sympy as sp
from sbmlmath.mathml_printer import SBMLMathMLPrinter

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
    value: float | None = None,
    constant: bool | None = None,
) -> libsbml.Parameter:
    """Add a parameter to the SBML model."""
    param = model.createParameter()

    check(param.setId(id_))

    if value is not None:
        check(param.setValue(value))

    if constant is not None:
        check(param.setConstant(constant))

    return param


def set_initial_assignment(
    model: libsbml.Model,
    symbol_id: str,
    expr: sp.Expr,
    literals_dimensionless: bool = True,
) -> libsbml.InitialAssignment:
    """Set the initial assignment for the given symbol.

    Overwrites any existing initial assignment for that symbol.

    :param literals_dimensionless:
        Whether to annotate numeric literals as dimensionless.
        (`sbmlmath` can't read back, e.g., ``10^x`` with a dimensionless
        base.)
    """
    if (ia := model.getInitialAssignmentBySymbol(symbol_id)) is None:
        ia = model.createInitialAssignment()
        check(ia.setSymbol(symbol_id))
    mathml = SBMLMathMLPrinter(
        sbml_level=model.getLevel(),
        sbml_version=model.getVersion(),
        literals_dimensionless=literals_dimensionless,
    ).doprint(expr)
    if (ast := libsbml.readMathMLFromString(mathml)) is None:
        raise ValueError(f"Failed to convert `{expr}' to MathML.")
    check(ia.setMath(ast))
    return ia
