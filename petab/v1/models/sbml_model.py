"""Functions for handling SBML models"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from pathlib import Path

import libsbml
import sympy as sp
from sympy.abc import _clash

from ..._utils import _generate_path
from ..sbml import (
    get_sbml_model,
    is_sbml_consistent,
    load_sbml_from_string,
    write_sbml,
)
from . import MODEL_TYPE_SBML
from .model import Model

__all__ = ["SbmlModel"]


class SbmlModel(Model):
    """PEtab wrapper for SBML models"""

    type_id = MODEL_TYPE_SBML

    def __init__(
        self,
        sbml_model: libsbml.Model = None,
        sbml_reader: libsbml.SBMLReader = None,
        sbml_document: libsbml.SBMLDocument = None,
        model_id: str = None,
        rel_path: Path | str | None = None,
        base_path: str | Path | None = None,
    ):
        """Constructor.

        :param sbml_model: SBML model. Optional if `sbml_document` is given.
        :param sbml_reader: SBML reader. Optional.
        :param sbml_document: SBML document. Optional if `sbml_model` is given.
        :param model_id: Model ID. Defaults to the SBML model ID."""
        super().__init__()

        self.rel_path = rel_path
        self.base_path = base_path

        if sbml_model is None and sbml_document is None:
            raise ValueError(
                "Either sbml_model or sbml_document must be given."
            )

        if sbml_model is None:
            sbml_model = sbml_document.getModel()

        if sbml_document is None:
            sbml_document = sbml_model.getSBMLDocument()

        self.sbml_reader: libsbml.SBMLReader | None = sbml_reader
        self.sbml_document: libsbml.SBMLDocument | None = sbml_document
        self.sbml_model: libsbml.Model | None = sbml_model

        self._model_id = model_id or sbml_model.getIdAttribute()

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()

        # libsbml stuff cannot be serialized directly
        if self.sbml_model:
            state["sbml_string"] = self.to_sbml_str()

        exclude = ["sbml_reader", "sbml_document", "sbml_model"]
        for key in exclude:
            state.pop(key)

        return state

    def __setstate__(self, state):
        """Set state after unpickling"""
        # load SBML model from pickled string
        sbml_string = state.pop("sbml_string", None)
        if sbml_string:
            (
                self.sbml_reader,
                self.sbml_document,
                self.sbml_model,
            ) = load_sbml_from_string(sbml_string)

        self.__dict__.update(state)

    @staticmethod
    def from_file(
        filepath_or_buffer, model_id: str = None, base_path: str | Path = None
    ) -> SbmlModel:
        sbml_reader, sbml_document, sbml_model = get_sbml_model(
            _generate_path(filepath_or_buffer, base_path=base_path)
        )
        return SbmlModel(
            sbml_model=sbml_model,
            sbml_reader=sbml_reader,
            sbml_document=sbml_document,
            model_id=model_id,
            rel_path=filepath_or_buffer,
            base_path=base_path,
        )

    @staticmethod
    def from_string(sbml_string, model_id: str = None) -> SbmlModel:
        """Create SBML model from an SBML string.

        :param sbml_string: SBML model as string.
        :param model_id: Model ID. Defaults to the SBML model ID.
        """
        sbml_reader, sbml_document, sbml_model = load_sbml_from_string(
            sbml_string
        )

        if not model_id:
            model_id = sbml_model.getIdAttribute()

        return SbmlModel(
            sbml_model=sbml_model,
            sbml_reader=sbml_reader,
            sbml_document=sbml_document,
            model_id=model_id,
        )

    @staticmethod
    def from_antimony(ant_model: str | Path, **kwargs) -> SbmlModel:
        """Create SBML model from an Antimony model.

        Requires the `antimony` package (https://github.com/sys-bio/antimony).

        :param ant_model: Antimony model as string or path to file.
            Strings are interpreted as Antimony model strings.
        :param kwargs: Additional keyword arguments passed to
            :meth:`SbmlModel.from_string`.
        """
        sbml_str = antimony2sbml(ant_model)
        return SbmlModel.from_string(sbml_str, **kwargs)

    def to_antimony(self) -> str:
        """Convert the SBML model to an Antimony string."""
        import antimony as ant

        sbml_str = self.to_sbml_str()

        ant.clearPreviousLoads()
        ant.freeAll()

        if ant.loadSBMLString(sbml_str) < 0:
            raise RuntimeError(ant.getLastError())

        return ant.getAntimonyString()

    def to_sbml_str(self) -> str:
        """Convert the SBML model to an SBML/XML string."""
        sbml_document = self.sbml_model.getSBMLDocument()
        sbml_writer = libsbml.SBMLWriter()
        return sbml_writer.writeSBMLToString(sbml_document)

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, model_id):
        self._model_id = model_id

    def to_file(self, filename: str | Path | None = None) -> None:
        write_sbml(
            self.sbml_document or self.sbml_model.getSBMLDocument(),
            filename or _generate_path(self.rel_path, self.base_path),
        )

    def get_parameter_value(self, id_: str) -> float:
        parameter = self.sbml_model.getParameter(id_)
        if not parameter:
            raise ValueError(f"Parameter {id_} does not exist.")
        return parameter.getValue()

    def get_free_parameter_ids_with_values(
        self,
    ) -> Iterable[tuple[str, float]]:
        rule_targets = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        def get_initial(p):
            # return the initial assignment value if there is one, and it is a
            # number; `None`, if there is a non-numeric initial assignment;
            # otherwise, the parameter value
            if ia := self.sbml_model.getInitialAssignmentBySymbol(p.getId()):
                sym_expr = sympify_sbml(ia.getMath())
                return (
                    float(sym_expr.evalf())
                    if sym_expr.evalf().is_Number
                    else None
                )
            return p.getValue()

        return (
            (p.getId(), initial)
            for p in self.sbml_model.getListOfParameters()
            if p.getId() not in rule_targets
            and (initial := get_initial(p)) is not None
        )

    def get_parameter_ids(self) -> Iterable[str]:
        rule_targets = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (
            p.getId()
            for p in self.sbml_model.getListOfParameters()
            if p.getId() not in rule_targets
        )

    def get_parameter_ids_with_values(self) -> Iterable[tuple[str, float]]:
        rule_targets = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (
            (p.getId(), p.getValue())
            for p in self.sbml_model.getListOfParameters()
            if p.getId() not in rule_targets
        )

    def has_entity_with_id(self, entity_id) -> bool:
        return self.sbml_model.getElementBySId(entity_id) is not None

    def get_valid_parameters_for_parameter_table(self) -> Iterable[str]:
        # All parameters except rule-targets
        disallowed_set = {
            ar.getVariable() for ar in self.sbml_model.getListOfRules()
        }

        return (
            p.getId()
            for p in self.sbml_model.getListOfParameters()
            if p.getId() not in disallowed_set
        )

    def get_valid_ids_for_condition_table(self) -> Iterable[str]:
        return (
            x.getId()
            for x in itertools.chain(
                self.sbml_model.getListOfParameters(),
                self.sbml_model.getListOfSpecies(),
                self.sbml_model.getListOfCompartments(),
            )
        )

    def symbol_allowed_in_observable_formula(self, id_: str) -> bool:
        return self.sbml_model.getElementBySId(id_) or id_ == "time"

    def is_valid(self) -> bool:
        return is_sbml_consistent(self.sbml_model.getSBMLDocument())

    def is_state_variable(self, id_: str) -> bool:
        return (
            self.sbml_model.getSpecies(id_) is not None
            or self.sbml_model.getCompartment(id_) is not None
            or self.sbml_model.getRuleByVariable(id_) is not None
        )


def sympify_sbml(sbml_obj: libsbml.ASTNode | libsbml.SBase) -> sp.Expr:
    """Convert SBML math expression to sympy expression.

    Parameters
    ----------
    sbml_obj:
        SBML math element or an SBML object with a math element.

    Returns
    -------
    The sympy expression corresponding to ``sbml_obj``.
    """
    ast_node = (
        sbml_obj
        if isinstance(sbml_obj, libsbml.ASTNode)
        else sbml_obj.getMath()
    )

    parser_settings = libsbml.L3ParserSettings(
        ast_node.getParentSBMLObject().getModel(),
        libsbml.L3P_PARSE_LOG_AS_LOG10,
        libsbml.L3P_EXPAND_UNARY_MINUS,
        libsbml.L3P_NO_UNITS,
        libsbml.L3P_AVOGADRO_IS_CSYMBOL,
        libsbml.L3P_COMPARE_BUILTINS_CASE_INSENSITIVE,
        None,
        libsbml.L3P_MODULO_IS_PIECEWISE,
    )

    formula_str = libsbml.formulaToL3StringWithSettings(
        ast_node, parser_settings
    )

    return sp.sympify(formula_str, locals=_clash)


def antimony2sbml(ant_model: str | Path) -> str:
    """Convert Antimony model to SBML.

    :param ant_model: Antimony model as string or path to file.
        Strings are interpreted as Antimony model strings.

    :returns:
        The SBML model as string.
    """
    import antimony as ant

    # Unload everything / free memory
    ant.clearPreviousLoads()
    ant.freeAll()

    try:
        # potentially fails because of too long file name
        is_file = ant_model and Path(ant_model).exists()
    except OSError:
        is_file = False

    if is_file:
        status = ant.loadAntimonyFile(str(ant_model))
    else:
        status = ant.loadAntimonyString(ant_model)
    if status < 0:
        raise RuntimeError(
            f"Antimony model could not be loaded: {ant.getLastError()}"
        )

    if (main_module_name := ant.getMainModuleName()) is None:
        raise AssertionError("There is no Antimony module.")

    sbml_str = ant.getSBMLString(main_module_name)
    if not sbml_str:
        raise ValueError("Antimony model could not be converted to SBML.")

    return sbml_str
