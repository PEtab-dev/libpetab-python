"""Functions for handling SBML models"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from functools import cached_property
from math import isnan as math_isnan
from pathlib import Path

import libsbml
import sympy as sp
from sbmlmath import sbml_math_to_sympy

from .._utils import _generate_path
from ..v1.sbml import (
    get_sbml_model,
    is_sbml_consistent,
    load_sbml_from_string,
    write_sbml,
)
from . import MODEL_TYPE_SBML
from ._entities import (
    Dynamics,
    EntityCategory,
    EntityInfo,
    Initializer,
    SpeciesQuantity,
    _matches,
)
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
        model_id: str | None = None,
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
        filepath_or_buffer,
        model_id: str | None = None,
        base_path: str | Path | None = None,
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
    def from_string(sbml_string, model_id: str | None = None) -> SbmlModel:
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

    def get_referenced_ids(self) -> set[str]:
        """Get the IDs that the model refers to.

        These are the IDs in the math of any model element (e.g., kinetic
        laws, rules, initial assignments, events), and those referred to as
        conversion factors. IDs that are only defined, but not referred to,
        are not included.
        """
        ids = set()
        for element in self.sbml_model.getListOfAllElements():
            if not (hasattr(element, "isSetMath") and element.isSetMath()):
                continue
            nodes = [element.getMath()]
            while nodes:
                node = nodes.pop()
                if node.getType() == libsbml.AST_NAME:
                    ids.add(node.getName())
                nodes += [
                    node.getChild(i) for i in range(node.getNumChildren())
                ]
        # conversion factors refer to parameters by attribute, not in math
        ids |= {
            element.getConversionFactor()
            for element in [
                self.sbml_model,
                *self.sbml_model.getListOfSpecies(),
            ]
            if element.isSetConversionFactor()
        }
        return ids

    def is_valid(self) -> bool:
        return is_sbml_consistent(self.sbml_model.getSBMLDocument())

    def is_state_variable(self, id_: str) -> bool:
        return (
            self.sbml_model.getSpecies(id_) is not None
            or self.sbml_model.getCompartment(id_) is not None
            or self.sbml_model.getRuleByVariable(id_) is not None
        )

    def _get_entity(self, id_: str) -> EntityInfo | None:
        """Get the description of the model entity with the given ID.

        Elements are classified as follows:

        * parameters, species, compartments, and species references
          (stoichiometries): by their rules, event assignments, and
          participation in reactions;
        * reactions (their rates): read-only quantities;
        * events and function definitions: non-quantities.

        Other elements with an ID, e.g., local parameters, unit definitions,
        or elements of SBML packages, are not entities.
        Of ``comp`` models, only the main model is considered.
        Results reflect the current state of the SBML model.

        See :meth:`petab.v2.models.Model._get_entity`.
        """
        element = self.sbml_model.getElementBySId(id_)
        if (
            element is None
            or element.getPackageName() != "core"
            or element.getTypeCode() not in _SBML_ENTITY_TYPES
        ):
            return None
        return _classify_sbml_element(
            element, _SbmlModelIndex(self.sbml_model, single=True)
        )

    def _iter_entities(
        self,
        category: EntityCategory | None = None,
        dynamics: Dynamics | None = None,
    ) -> Iterator[EntityInfo]:
        """Iterate over the entities of the model.

        See :meth:`_get_entity` and
        :meth:`petab.v2.models.Model._iter_entities`.
        """
        index = _SbmlModelIndex(self.sbml_model)
        # Not `getListOfAllElements`, which is quadratic in the model size,
        #  and which would include local parameters of SBML L2 models.
        elements = [
            element
            for element in _iter_sbml_elements(self.sbml_model)
            if element.isSetIdAttribute()
        ]

        def generate():
            for element in elements:
                info = _classify_sbml_element(element, index)
                if info is not None and _matches(info, category, dynamics):
                    yield info

        return generate()


def _iter_sbml_elements(model: libsbml.Model) -> Iterator[libsbml.SBase]:
    """Iterate over the SBML core elements of a model that may have an ID.

    Local parameters are not included.
    """
    # Not rules, initial assignments, or event assignments: libsbml does not
    #  resolve their `id` attribute in `getElementBySId`.
    yield from model.getListOfFunctionDefinitions()
    yield from model.getListOfCompartments()
    yield from model.getListOfSpecies()
    yield from model.getListOfParameters()
    for reaction in model.getListOfReactions():
        yield reaction
        yield from reaction.getListOfReactants()
        yield from reaction.getListOfProducts()
        yield from reaction.getListOfModifiers()
    yield from model.getListOfEvents()


def _iter_sbml_math(model: libsbml.Model) -> Iterator[libsbml.ASTNode]:
    """Iterate over the math of all SBML core elements of a model."""
    for element in itertools.chain(
        model.getListOfFunctionDefinitions(),
        model.getListOfInitialAssignments(),
        model.getListOfRules(),
        model.getListOfConstraints(),
        (
            reaction.getKineticLaw()
            for reaction in model.getListOfReactions()
            if reaction.isSetKineticLaw()
        ),
        *(
            (
                event.getTrigger(),
                event.getDelay(),
                event.getPriority(),
                *event.getListOfEventAssignments(),
            )
            for event in model.getListOfEvents()
        ),
    ):
        if element is not None and element.isSetMath():
            yield element.getMath()


#: Type codes of SBML elements whose value can be set or computed.
_SBML_VARIABLE_TYPES = {
    libsbml.SBML_PARAMETER,
    libsbml.SBML_SPECIES,
    libsbml.SBML_COMPARTMENT,
    libsbml.SBML_SPECIES_REFERENCE,
}

#: Type codes of the SBML core elements that are entities
#: (see :func:`_iter_sbml_elements`).
_SBML_ENTITY_TYPES = {
    *_SBML_VARIABLE_TYPES,
    libsbml.SBML_FUNCTION_DEFINITION,
    libsbml.SBML_REACTION,
    libsbml.SBML_MODIFIER_SPECIES_REFERENCE,
    libsbml.SBML_EVENT,
}


class _SbmlModelIndex:
    """Model-wide information needed to classify SBML elements.

    In bulk mode, each aggregate is computed on first use, by one pass over
    the model. In single mode (for classifying a single element), targeted
    lookups are used where possible.
    Only valid as long as the model is not modified.
    """

    def __init__(self, model: libsbml.Model, single: bool = False):
        self.model = model
        self.single = single

    def rule_type(self, id_: str) -> int | None:
        """The type code of the assignment or rate rule for the given ID."""
        if not self.single:
            return self.rule_types.get(id_)
        rule = self.model.getRuleByVariable(id_)
        if rule is None or rule.isAlgebraic():
            return None
        return rule.getTypeCode()

    def initial_assignment(self, id_: str) -> libsbml.InitialAssignment | None:
        if not self.single:
            return self.initial_assignments.get(id_)
        return self.model.getInitialAssignmentBySymbol(id_)

    def is_event_assigned(self, id_: str) -> bool:
        if not self.single:
            return id_ in self.event_assignment_targets
        return any(
            ea.getVariable() == id_
            for event in self.model.getListOfEvents()
            for ea in event.getListOfEventAssignments()
        )

    def is_reactant_or_product(self, id_: str) -> bool:
        if not self.single:
            return id_ in self.reactants_and_products
        return any(
            species_ref.getSpecies() == id_
            for reaction in self.model.getListOfReactions()
            for species_ref in itertools.chain(
                reaction.getListOfReactants(), reaction.getListOfProducts()
            )
        )

    def algebraic_info(self, id_: str) -> tuple[bool, list[str]]:
        """Whether the entity is determined by an algebraic rule, and notes.

        See :attr:`algebraic`.
        """
        if self.single and not any(
            node.getType() == libsbml.AST_NAME and node.getName() == id_
            for rule in self.model.getListOfRules()
            if rule.isAlgebraic() and rule.isSetMath()
            for node in _iter_ast(rule.getMath())
        ):
            return False, []
        determined, notes = self.algebraic
        return id_ in determined, notes.get(id_, [])

    @cached_property
    def rule_types(self) -> dict[str, int]:
        """Rule type codes by rule variable."""
        return {
            rule.getVariable(): rule.getTypeCode()
            for rule in self.model.getListOfRules()
            if not rule.isAlgebraic()
        }

    @cached_property
    def event_assignment_targets(self) -> set[str]:
        return {
            ea.getVariable()
            for event in self.model.getListOfEvents()
            for ea in event.getListOfEventAssignments()
        }

    @cached_property
    def initial_assignments(self) -> dict[str, libsbml.InitialAssignment]:
        return {
            ia.getSymbol(): ia
            for ia in self.model.getListOfInitialAssignments()
        }

    @cached_property
    def reactants_and_products(self) -> set[str]:
        """IDs of species that are reactants or products of any reaction."""
        return {
            species_ref.getSpecies()
            for reaction in self.model.getListOfReactions()
            for species_ref in itertools.chain(
                reaction.getListOfReactants(), reaction.getListOfProducts()
            )
        }

    @cached_property
    def algebraic(self) -> tuple[set[str], dict[str, list[str]]]:
        """Entities determined by algebraic rules.

        :returns: The IDs of entities that are the only candidate for being
            determined by some algebraic rule, and notes for entities that
            are one of several candidates.
        """
        rules = [
            rule for rule in self.model.getListOfRules() if rule.isAlgebraic()
        ]
        if not rules:
            return set(), {}

        rate_of_targets = {
            node.getChild(0).getName()
            for math in _iter_sbml_math(self.model)
            for node in _iter_ast(math)
            if node.getType() == libsbml.AST_FUNCTION_RATE_OF
            and node.getNumChildren() == 1
        }

        def is_candidate(id_: str) -> bool:
            element = self.model.getElementBySId(id_)
            if (
                element is None
                or element.getTypeCode() not in _SBML_VARIABLE_TYPES
                or element.getConstant()
                or self.rule_type(id_) is not None
                or id_ in rate_of_targets
            ):
                return False
            return not (
                element.getTypeCode() == libsbml.SBML_SPECIES
                and not element.getBoundaryCondition()
                and self.is_reactant_or_product(id_)
            )

        # (rule formula, candidates) for each rule not yet resolved
        undecided = [
            (
                libsbml.formulaToL3String(rule.getMath()),
                [
                    c
                    for c in sorted(
                        {
                            node.getName()
                            for node in _iter_ast(rule.getMath())
                            if node.getType() == libsbml.AST_NAME
                        }
                    )
                    if is_candidate(c)
                ],
            )
            for rule in rules
            if rule.isSetMath()
        ]
        # A rule with a single undetermined candidate determines it;
        #  repeat until no rule determines a new entity.
        determined = set()
        changed = True
        while changed:
            changed = False
            still_undecided = []
            for formula, candidates in undecided:
                remaining = [c for c in candidates if c not in determined]
                if len(remaining) <= 1:
                    determined.update(remaining)
                    changed = True
                else:
                    still_undecided.append((formula, remaining))
            undecided = still_undecided

        notes = {}
        for formula, candidates in undecided:
            for candidate in candidates:
                notes.setdefault(candidate, []).append(
                    f"It may be determined by the algebraic rule "
                    f"`0 = {formula}`, which has several candidate "
                    f"variables ({', '.join(candidates)})."
                )
        return determined, notes


def _iter_ast(node: libsbml.ASTNode) -> Iterator[libsbml.ASTNode]:
    """Iterate over all nodes of an AST."""
    nodes = [node]
    while nodes:
        node = nodes.pop()
        yield node
        nodes.extend(node.getChild(i) for i in range(node.getNumChildren()))


#: AST node types that refer to the model state or time.
_AST_NON_CLOSED_TYPES = {
    libsbml.AST_NAME,
    libsbml.AST_NAME_TIME,
    libsbml.AST_FUNCTION,
    libsbml.AST_FUNCTION_DELAY,
    libsbml.AST_FUNCTION_RATE_OF,
}


def _initializer_from_math(math: libsbml.ASTNode) -> Initializer:
    """Classify an initial-assignment formula.

    Formulas over numbers only are evaluated; anything referring to model
    entities, time, or function definitions is ``FORMULA``.
    """
    dependencies = set()
    closed = True
    for node in _iter_ast(math):
        if node.getType() in _AST_NON_CLOSED_TYPES:
            closed = False
            if node.getType() in (libsbml.AST_NAME, libsbml.AST_FUNCTION):
                dependencies.add(node.getName())

    if closed:
        value = libsbml.SBMLTransforms.evaluateASTNode(math)
        # libsbml returns NaN for anything it cannot evaluate
        if not math_isnan(value) or math.isNaN():
            return Initializer.literal(value)

    math = math.deepCopy()
    return Initializer.formula(
        raw=libsbml.formulaToL3String(math),
        dependencies=dependencies,
        expression=lambda: sympify_sbml(math),
    )


def _formula_initializer(
    formula: str, dependencies: set[str], model: libsbml.Model
) -> Initializer:
    # with the model, so that IDs like `pi` or `time` are not taken as
    #  constants or csymbols
    math = libsbml.parseL3FormulaWithModel(formula, model)
    return Initializer.formula(
        raw=formula,
        dependencies=dependencies,
        expression=lambda: sympify_sbml(math),
    )


def _attribute_initializer(element: libsbml.SBase) -> Initializer:
    """The initial value given by the attributes of an element."""
    type_code = element.getTypeCode()
    if type_code == libsbml.SBML_PARAMETER:
        if element.isSetValue():
            return Initializer.literal(element.getValue())
    elif type_code == libsbml.SBML_COMPARTMENT:
        if element.isSetSize():
            return Initializer.literal(element.getSize())
    elif type_code == libsbml.SBML_SPECIES_REFERENCE:
        # defaults to 1 before SBML L3
        if element.isSetStoichiometry() or element.getLevel() < 3:
            return Initializer.literal(element.getStoichiometry())
    elif type_code == libsbml.SBML_SPECIES:
        compartment = element.getCompartment()
        if element.getHasOnlySubstanceUnits():
            if element.isSetInitialAmount():
                return Initializer.literal(element.getInitialAmount())
            if element.isSetInitialConcentration():
                return _formula_initializer(
                    f"{element.getInitialConcentration()!r} * {compartment}",
                    {compartment},
                    element.getModel(),
                )
        else:
            if element.isSetInitialConcentration():
                return Initializer.literal(element.getInitialConcentration())
            if element.isSetInitialAmount():
                return _formula_initializer(
                    f"{element.getInitialAmount()!r} / {compartment}",
                    {compartment},
                    element.getModel(),
                )
    return Initializer.unspecified()


def _classify_sbml_element(
    element: libsbml.SBase, index: _SbmlModelIndex
) -> EntityInfo | None:
    """Classify an SBML element with an ID."""
    type_code = element.getTypeCode()
    id_ = element.getIdAttribute()
    native_type = f"sbml:{type(element).__name__}"

    if type_code == libsbml.SBML_REACTION:
        return EntityInfo(
            id=id_,
            category=EntityCategory.OTHER_QUANTITY,
            dynamics=Dynamics.ALGEBRAIC,
            native_type=native_type,
        )

    if type_code not in _SBML_VARIABLE_TYPES:
        return EntityInfo(
            id=id_,
            category=EntityCategory.NON_QUANTITY,
            dynamics=None,
            usable_in_math=False,
            native_type=native_type,
        )

    category = {
        libsbml.SBML_PARAMETER: EntityCategory.PARAMETER,
        libsbml.SBML_SPECIES: EntityCategory.SPECIES,
        libsbml.SBML_COMPARTMENT: EntityCategory.COMPARTMENT,
        libsbml.SBML_SPECIES_REFERENCE: EntityCategory.OTHER_QUANTITY,
    }[type_code]

    rule_type = index.rule_type(id_)
    algebraic_determined, algebraic_notes = (
        index.algebraic_info(id_) if rule_type is None else (False, [])
    )
    if (
        rule_type == libsbml.SBML_ASSIGNMENT_RULE
        or algebraic_determined
        or (
            type_code == libsbml.SBML_SPECIES_REFERENCE
            and element.isSetStoichiometryMath()
        )
    ):
        dynamics = Dynamics.ALGEBRAIC
    elif rule_type == libsbml.SBML_RATE_RULE or (
        type_code == libsbml.SBML_SPECIES
        and not element.getBoundaryCondition()
        and not element.getConstant()
        and index.is_reactant_or_product(id_)
    ):
        dynamics = Dynamics.DIFFERENTIAL
    else:
        dynamics = Dynamics.CONSTANT

    if dynamics == Dynamics.ALGEBRAIC:
        initializer = Initializer.unspecified()
    elif (ia := index.initial_assignment(id_)) is not None and ia.isSetMath():
        initializer = _initializer_from_math(ia.getMath())
    else:
        initializer = _attribute_initializer(element)

    species_quantity = None
    if type_code == libsbml.SBML_SPECIES:
        species_quantity = (
            SpeciesQuantity.AMOUNT
            if element.getHasOnlySubstanceUnits()
            else SpeciesQuantity.CONCENTRATION
        )

    return EntityInfo(
        id=id_,
        category=category,
        dynamics=dynamics,
        event_assigned=index.is_event_assigned(id_),
        initializer=initializer,
        # species reference IDs only have a mathematical meaning since L3
        usable_in_math=element.getLevel() >= 3
        or type_code != libsbml.SBML_SPECIES_REFERENCE,
        native_type=native_type,
        species_quantity=species_quantity,
        notes=tuple(algebraic_notes),
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
    return sbml_math_to_sympy(sbml_obj)


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
