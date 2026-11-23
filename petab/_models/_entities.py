"""Classification of model entities.

Each model wrapper (:class:`petab.v2.models.Model` subclass) describes the
entities of its model as :class:`EntityInfo` objects
(:meth:`petab.v2.models.Model._get_entity`,
:meth:`petab.v2.models.Model._iter_entities`).
The functions in this module answer PEtab-level questions about an entity,
e.g., whether it may be a condition target or a parameter-table row,
based on that description only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import sympy as sp

__all__ = [
    "Dynamics",
    "EligibilityIssue",
    "EntityCategory",
    "EntityInfo",
    "Initializer",
    "InitializerKind",
    "SpeciesQuantity",
    "condition_target_issue",
    "has_literal_default",
    "is_constant_parameter",
    "parameter_table_issue",
]


class EntityCategory(StrEnum):
    """What a model entity is."""

    #: A plain scalar quantity, e.g., an SBML parameter.
    PARAMETER = "parameter"
    #: An amount or concentration of a chemical species.
    SPECIES = "species"
    #: A compartment; its value is the compartment size.
    COMPARTMENT = "compartment"
    #: Any other quantity, e.g., a reaction rate, a stoichiometry, or an
    #: observable or expression defined in the model.
    #: Neither a valid condition target nor a parameter-table entry.
    OTHER_QUANTITY = "other_quantity"
    #: An identifier that does not denote a number, e.g., an event,
    #: a function definition, or a molecule type.
    NON_QUANTITY = "non_quantity"


class Dynamics(StrEnum):
    """How the value of an entity evolves over time.

    This is the *entity type* defined by the PEtab v2 specification.
    """

    #: Constant, or piecewise constant if subject to event assignments.
    CONSTANT = "constant"
    #: Defined by a time derivative.
    DIFFERENTIAL = "differential"
    #: Determined by an algebraic equation that is in effect throughout the
    #: simulation, e.g., the target of an SBML assignment rule, the
    #: unknown of an SBML algebraic rule, a reaction rate, or a PySB
    #: expression.
    ALGEBRAIC = "algebraic"


class SpeciesQuantity(StrEnum):
    """Whether the value of a species is an amount or a concentration."""

    AMOUNT = "amount"
    CONCENTRATION = "concentration"


class InitializerKind(StrEnum):
    """How the model defines the initial value of an entity."""

    #: A number, or an expression over numbers only.
    LITERAL = "literal"
    #: An expression over other model entities.
    FORMULA = "formula"
    #: No initial value, or not introspectable.
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True, kw_only=True)
class Initializer:
    """The initial value of an entity, as defined by the model itself.

    This is the value before PEtab applies any parameter-table values or
    condition changes.
    Use :meth:`literal`, :meth:`formula`, or :meth:`unspecified` to create
    instances.

    :ivar kind: How the initial value is defined.
    :ivar value: The initial value, for ``LITERAL`` initializers only.
    :ivar raw: The defining expression in the model's own syntax,
        for ``FORMULA`` initializers only.
    :ivar dependencies: IDs of the model entities and functions that
        the defining expression refers to, for ``FORMULA`` initializers only.
    """

    kind: InitializerKind
    value: float | None = None
    raw: str | None = None
    dependencies: frozenset[str] = frozenset()
    _expression_factory: Callable[[], sp.Expr] | None = field(
        default=None, compare=False, repr=False
    )

    @classmethod
    def literal(cls, value: float) -> Initializer:
        """Create an initializer for a numeric initial value."""
        return cls(kind=InitializerKind.LITERAL, value=float(value))

    @classmethod
    def formula(
        cls,
        raw: str,
        dependencies: Iterable[str] = (),
        expression: Callable[[], sp.Expr] | None = None,
    ) -> Initializer:
        """Create an initializer for an initial value defined by an
        expression.

        :param raw: The expression in the model's own syntax.
        :param dependencies: IDs the expression refers to.
        :param expression: A function returning the expression as a sympy
            expression; called at most once, when :attr:`expression` is
            first accessed.
        """
        return cls(
            kind=InitializerKind.FORMULA,
            raw=raw,
            dependencies=frozenset(dependencies),
            _expression_factory=expression,
        )

    @classmethod
    def unspecified(cls) -> Initializer:
        """Create an initializer for an entity without initial value."""
        return cls(kind=InitializerKind.UNSPECIFIED)

    @cached_property
    def expression(self) -> sp.Expr | None:
        """The defining expression as a sympy expression.

        ``None`` for non-``FORMULA`` initializers, and if the model type
        does not support conversion to sympy.
        """
        if self._expression_factory is None:
            return None
        return self._expression_factory()


@dataclass(frozen=True, kw_only=True)
class EntityInfo:
    """Description of a model entity.

    :ivar id: The entity's ID in the model, or, for PySB and BNGL species,
        the species pattern string.
    :ivar category: What the entity is.
    :ivar dynamics: How the entity's value evolves.
        ``None`` for :attr:`EntityCategory.NON_QUANTITY` entities.
    :ivar event_assigned: Whether the entity is the target of an event
        assignment. A ``CONSTANT`` entity that is not event-assigned is
        *truly constant* in the sense of the PEtab v2 specification.
    :ivar initializer: How the model defines the entity's initial value.
    :ivar usable_in_math: Whether the ID may be used as a symbol in PEtab
        math expressions, e.g., in observable formulas.
    :ivar native_type: The model-specific type of the entity,
        e.g., ``"sbml:Species"``. For messages only.
    :ivar species_quantity: Whether the value is an amount or a
        concentration. For species only.
    :ivar declared: Whether the model declares the entity.
        ``False`` for PySB and BNGL species patterns that are valid, but
        not declared by the model; the model's rules may or may not
        generate such a species.
    :ivar notes: Remarks about questionable aspects of the entity,
        e.g., an ambiguous algebraic rule. Validation reports each as a
        warning.
    """

    id: str
    category: EntityCategory
    dynamics: Dynamics | None
    event_assigned: bool = False
    initializer: Initializer = field(default_factory=Initializer.unspecified)
    usable_in_math: bool = True
    native_type: str = ""
    species_quantity: SpeciesQuantity | None = None
    declared: bool = True
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EligibilityIssue:
    """A reason why an entity is not, or not unconditionally, eligible for
    some use.

    :ivar message: Human-readable description, naming the model construct.
    :ivar level: ``"error"`` if the use is invalid, ``"warning"`` if it is
        allowed but questionable.
    """

    message: str
    level: Literal["error", "warning"]


def is_constant_parameter(info: EntityInfo) -> bool:
    """Check whether the entity is a parameter of constant type.

    Event-assigned parameters are included.
    """
    return (
        info.category == EntityCategory.PARAMETER
        and info.dynamics == Dynamics.CONSTANT
    )


def has_literal_default(info: EntityInfo) -> bool:
    """Check whether the model defines a numeric initial value for the
    entity."""
    return info.initializer.kind == InitializerKind.LITERAL


def _matches(
    info: EntityInfo,
    category: EntityCategory | None,
    dynamics: Dynamics | None,
) -> bool:
    """Check whether the entity matches the given filters."""
    return (category is None or info.category == category) and (
        dynamics is None or info.dynamics == dynamics
    )


def _describe(info: EntityInfo) -> str:
    """Short description of an entity for messages."""
    if info.native_type:
        return f"`{info.id}` ({info.native_type})"
    return f"`{info.id}`"


def _notes_issue(info: EntityInfo) -> EligibilityIssue | None:
    if not info.notes:
        return None
    return EligibilityIssue(
        f"{_describe(info)}: " + " ".join(info.notes), "warning"
    )


def _check_petab_version(petab_version: int) -> None:
    if petab_version not in (1, 2):
        raise ValueError(f"Unsupported PEtab version: {petab_version}")


def condition_target_issue(
    info: EntityInfo, *, petab_version: Literal[1, 2]
) -> EligibilityIssue | None:
    """Check whether the entity may be changed by the condition table.

    :param info: The entity.
    :param petab_version: The PEtab major version whose rules apply.
    :returns: ``None`` if the entity is a valid condition target without
        reservations, otherwise the most severe issue.
    """
    _check_petab_version(petab_version)
    what = _describe(info)

    if info.category not in (
        EntityCategory.PARAMETER,
        EntityCategory.SPECIES,
        EntityCategory.COMPARTMENT,
    ):
        return EligibilityIssue(
            f"{what} cannot be changed by a condition: it is not a "
            "parameter, species, or compartment.",
            "error",
        )

    if info.dynamics == Dynamics.ALGEBRAIC:
        if petab_version == 2:
            return EligibilityIssue(
                f"{what} is defined by an algebraic assignment (e.g., an "
                "assignment rule) and cannot be changed by a condition.",
                "error",
            )
        return EligibilityIssue(
            f"{what} is defined by an algebraic assignment (e.g., an "
            "assignment rule). The assignment overrides any condition "
            "value, so the condition has no effect.",
            "warning",
        )

    # PEtab v2 specifies that the condition value replaces the model's
    #  definition, PEtab v1 does not
    if (
        petab_version == 1
        and info.category == EntityCategory.PARAMETER
        and info.initializer.kind == InitializerKind.FORMULA
    ):
        return EligibilityIssue(
            f"{what} is defined in the model as `{info.initializer.raw}`. "
            "The condition value replaces that definition; some "
            "simulators ignore it.",
            "warning",
        )

    if not info.declared:
        return EligibilityIssue(
            f"{what} is not declared in the model; it is a valid species "
            "pattern, but it may not exist in the generated reaction "
            "network.",
            "warning",
        )

    return _notes_issue(info)


def parameter_table_issue(
    info: EntityInfo, *, petab_version: Literal[1, 2]
) -> EligibilityIssue | None:
    """Check whether the entity may be listed in the parameter table.

    :param info: The entity.
    :param petab_version: The PEtab major version whose rules apply.
    :returns: ``None`` if the entity is a valid parameter-table entry
        without reservations, otherwise the most severe issue.
    """
    _check_petab_version(petab_version)
    what = _describe(info)

    if info.category != EntityCategory.PARAMETER:
        return EligibilityIssue(
            f"{what} cannot be listed in the parameter table: "
            "it is not a parameter.",
            "error",
        )

    if info.dynamics == Dynamics.ALGEBRAIC:
        return EligibilityIssue(
            f"{what} cannot be listed in the parameter table: it is "
            "defined by an algebraic assignment (e.g., an assignment rule).",
            "error",
        )

    if info.dynamics == Dynamics.DIFFERENTIAL:
        msg = (
            f"{what} cannot be listed in the parameter table: it is defined "
            "by a time derivative (e.g., a rate rule)."
        )
        if petab_version == 1:
            msg += " This is a restriction of this library."
        return EligibilityIssue(msg, "error")

    if info.event_assigned:
        if petab_version == 2:
            return EligibilityIssue(
                f"{what} cannot be listed in the parameter table: "
                "it is the target of an event assignment.",
                "error",
            )
        return EligibilityIssue(
            f"{what} is the target of an event assignment. This is allowed "
            "in PEtab v1, but invalid in PEtab v2.",
            "warning",
        )

    if info.initializer.kind == InitializerKind.FORMULA:
        return EligibilityIssue(
            f"{what} is defined in the model as `{info.initializer.raw}`. "
            "In PEtab v2, the parameter-table value replaces that "
            "definition; PEtab v1 does not specify this, and some "
            "simulators keep the model definition and ignore the "
            "parameter-table value.",
            "warning",
        )

    return _notes_issue(info)
