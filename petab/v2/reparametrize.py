"""Reparametrization of PEtab problems."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from itertools import chain
from typing import Literal

import libsbml
import sympy as sp

from . import C
from .core import Parameter, PriorDistribution, Problem
from .lint import get_placeholders
from .models._sbml_utils import add_sbml_parameter, check
from .models.sbml_model import SbmlModel

__all__ = ["reparametrize_log"]

#: x-space priors and the corresponding priors of log_b(x), given as
#:  (prior type, function(prior parameters, ln(b)) -> prior parameters)
_LOG_PRIOR_TO_THETA = {
    C.LOG_NORMAL: (C.NORMAL, lambda pars, ln_b: [p / ln_b for p in pars]),
    C.LOG_LAPLACE: (C.LAPLACE, lambda pars, ln_b: [p / ln_b for p in pars]),
    C.LOG_UNIFORM: (
        C.UNIFORM,
        lambda pars, ln_b: [math.log(p) / ln_b for p in pars],
    ),
}


def reparametrize_log(
    problem: Problem,
    parameter_ids: Iterable[str],
    *,
    base: Literal[10, "e"] = 10,
    priors: dict[str, tuple[str, list[float]] | None] | None = None,
) -> dict[str, str]:
    """Estimate parameters on log scale.

    For each parameter ``p``, replace ``p`` in the parameter table by a new
    parameter ``q = log_base(p)`` and define ``p = base^q``.
    This is the PEtab v2 way of estimating parameters on log scale
    (and of specifying priors for the log-transformed parameter).

    The problem is modified in place:

    * Parameter table: ``p`` is replaced by ``q`` with log-transformed bounds
      and nominal value.
      Parameters without a positive nominal value get no nominal value for
      ``q``.
    * Model: If ``p`` is a model parameter, ``q`` is added to the model,
      and ``p`` becomes the target of the assignment rule ``p = base^q``.
      Only SBML models are supported.
    * Observable and noise formulas, and condition table target values:
      ``p`` is substituted by ``base^q``.
    * Measurement table: Overrides ``p`` are replaced by ``q``. The respective
      placeholder ``ph`` is substituted by ``base^ph`` in the observable
      (and noise) formula. This requires that all overrides for that
      placeholder are reparametrized parameters or positive numbers.
      Numbers are log-transformed.

    The new ID is ``log10_<p>`` for base 10 and ``log_<p>`` for base e,
    with a numeric suffix if necessary to make it unique.

    :param problem:
        The PEtab problem to modify.
    :param parameter_ids:
        IDs of the parameters to reparametrize.
    :param base:
        The base of the logarithm.
    :param priors:
        Prior distributions of the new parameters as
        ``{p: (prior_distribution, prior_parameters)}``, or ``{p: None}``
        for no prior.
        For parameters not included here, the prior is derived from the prior
        of ``p``, where possible:
        ``log-normal(m, s)`` -> ``normal(m / ln(b), s / ln(b))``,
        ``log-laplace(m, s)`` -> ``laplace(m / ln(b), s / ln(b))``,
        ``log-uniform(a, b)`` -> ``uniform(log_b(a), log_b(b))``,
        no prior -> no prior (i.e., uniform on log scale).
        For non-estimated parameters, no prior is derived.
    :return:
        A dictionary mapping old to new parameter IDs.
    :raises ValueError:
        If a parameter cannot be reparametrized, e.g., because its lower bound
        is not positive, or if its prior cannot be expressed in terms of the
        new parameter.
    :raises NotImplementedError:
        If the problem has features that are not supported yet,
        e.g., multiple or non-SBML models, mapping table entries for the
        parameters, or placeholders with overrides that cannot be
        log-transformed.
    """
    parameter_ids = list(dict.fromkeys(parameter_ids))
    priors = priors or {}
    if base == 10:
        ln_b = math.log(10)
        b_expr = sp.Integer(10)
        prefix = "log10_"
    elif base == "e":
        ln_b = 1.0
        b_expr = sp.E
        prefix = "log_"
    else:
        raise ValueError(f"Unsupported base: {base}")

    if unknown := set(priors) - set(parameter_ids):
        raise ValueError(f"Priors given for unknown parameters: {unknown}")

    if not parameter_ids:
        return {}

    if problem.extensions.sciml is not None:
        raise NotImplementedError(
            "Reparametrization of PEtab SciML problems is not supported."
        )

    if len(problem.models) > 1:
        raise NotImplementedError(
            "Reparametrization of multi-model problems is not supported."
        )
    model = problem.model

    def log_b(x: float) -> float:
        return math.log(x) / ln_b

    new_parameters = {}
    model_entities = []
    for p in parameter_ids:
        par = _get_parameter(problem, p)
        if any(p in (m.petab_id, m.model_id) for m in problem.mappings):
            raise NotImplementedError(
                f"Parameter `{p}' occurs in the mapping table."
            )
        if model is not None and model.has_entity_with_id(p):
            if not isinstance(model, SbmlModel):
                raise NotImplementedError(
                    "Reparametrization of model parameters is only "
                    "supported for SBML models."
                )
            if model.sbml_model.getParameter(p) is None:
                raise NotImplementedError(
                    f"Model entity `{p}' is not an SBML parameter."
                )
            model_entities.append(p)
        if any(bound is not None and bound <= 0 for bound in (par.lb, par.ub)):
            # (bounds are required for estimated parameters)
            raise ValueError(
                f"Bounds of parameter `{p}' must be positive, "
                f"but got [{par.lb}, {par.ub}]."
            )

        if (
            par.nominal_value is None
            or par.nominal_value == "array"
            or par.nominal_value <= 0
        ):
            if not par.estimate:
                raise ValueError(
                    f"Non-estimated parameter `{p}' must have a positive "
                    f"nominal value, but got {par.nominal_value}."
                )
            warnings.warn(
                f"Parameter `{p}' has no positive nominal value "
                f"({par.nominal_value}). The nominal value of its "
                "reparametrization is left empty.",
                stacklevel=2,
            )
            nominal_value = None
        else:
            nominal_value = log_b(par.nominal_value)

        if p in priors and priors[p] is not None:
            prior_distribution, prior_parameters = priors[p]
            prior_distribution = PriorDistribution(prior_distribution)
        elif p in priors or par.prior_distribution is None or not par.estimate:
            # priors of non-estimated parameters are irrelevant
            prior_distribution, prior_parameters = None, []
        elif par.prior_distribution in _LOG_PRIOR_TO_THETA:
            prior_distribution, f = _LOG_PRIOR_TO_THETA[par.prior_distribution]
            prior_parameters = f(par.prior_parameters, ln_b)
        else:
            raise ValueError(
                f"Prior `{par.prior_distribution}' of parameter `{p}' "
                "cannot be expressed in terms of the log-transformed "
                "parameter."
            )

        new_parameters[p] = Parameter(
            id=p,
            name=par.name and f"{prefix.removesuffix('_')}({par.name})",
            lb=None if par.lb is None else log_b(par.lb),
            ub=None if par.ub is None else log_b(par.ub),
            nominal_value=nominal_value,
            estimate=par.estimate,
            prior_distribution=prior_distribution,
            prior_parameters=prior_parameters,
            **(par.model_extra or {}),
        )

    # assign new IDs
    used_ids = _get_used_ids(problem)
    id_map = {}
    for p in parameter_ids:
        q = f"{prefix}{p}"
        i = 0
        while q in used_ids or (
            model is not None and model.has_entity_with_id(q)
        ):
            i += 1
            q = f"{prefix}{p}_{i}"
        used_ids.add(q)
        id_map[p] = q
        new_parameters[p].id = q

    _reparametrize_overrides(problem, id_map, b_expr, log_b)

    # substitute p -> b^q in all expressions
    def subs(expr: sp.Basic | None) -> sp.Basic | None:
        if expr is None:
            return expr
        return expr.subs(
            {
                s: b_expr ** sp.Symbol(id_map[s.name], real=True)
                for s in expr.free_symbols
                if isinstance(s, sp.Symbol) and s.name in id_map
            }
        )

    for observable in problem.observables:
        observable.formula = subs(observable.formula)
        observable.noise_formula = subs(observable.noise_formula)
    for condition in problem.conditions:
        for change in condition.changes:
            change.target_value = subs(change.target_value)

    # update parameter tables
    for table in problem.parameter_tables:
        table.elements = [
            new_parameters.get(par.id, par) for par in table.elements
        ]

    # update the model
    for p in model_entities:
        _add_sbml_log_rule(
            model.sbml_model,
            p,
            id_map[p],
            b_expr,
            new_parameters[p].nominal_value,
        )

    return id_map


def _get_parameter(problem: Problem, parameter_id: str) -> Parameter:
    """Get the parameter table entry for the given ID."""
    for par in problem.parameters:
        if par.id == parameter_id:
            return par
    raise ValueError(
        f"Parameter `{parameter_id}' is not in the parameter table."
    )


def _get_used_ids(problem: Problem) -> set[str]:
    """Get all IDs and symbols used in the PEtab tables."""
    ids = set(get_placeholders(problem))
    ids |= {p.id for p in problem.parameters}
    ids |= {o.id for o in problem.observables}
    ids |= {c.id for c in problem.conditions}
    ids |= {e.id for e in problem.experiments}
    ids |= {m.petab_id for m in problem.mappings}
    ids |= {m.model_id for m in problem.mappings if m.model_id}
    ids |= {
        change.target_id
        for cond in problem.conditions
        for change in cond.changes
    }
    expressions = chain(
        (o.formula for o in problem.observables),
        (o.noise_formula for o in problem.observables),
        (
            change.target_value
            for cond in problem.conditions
            for change in cond.changes
        ),
        chain.from_iterable(
            chain(m.observable_parameters, m.noise_parameters)
            for m in problem.measurements
        ),
    )
    ids |= {
        str(s)
        for expr in expressions
        if expr is not None
        for s in expr.free_symbols
    }
    return ids


def _get_override_slots(
    problem: Problem,
) -> dict[tuple[str, str, int], set[sp.Basic]]:
    """Get the values of all placeholder overrides.

    :return:
        Dictionary mapping ``(observable ID, placeholder type, index)``
        to the set of override values. The placeholder type is
        ``"observable"`` or ``"noise"``.
    """
    slots = {}
    for m in problem.measurements:
        for type_, overrides in (
            ("observable", m.observable_parameters),
            ("noise", m.noise_parameters),
        ):
            for i, value in enumerate(overrides):
                slots.setdefault((m.observable_id, type_, i), set()).add(value)
    return slots


def _reparametrize_overrides(
    problem: Problem,
    id_map: dict[str, str],
    b_expr: sp.Expr,
    log_b,
) -> None:
    """Log-transform placeholders overridden by reparametrized parameters."""
    observables = {o.id: o for o in problem.observables}

    def get_placeholder(obs_id: str, type_: str, i: int) -> sp.Symbol:
        observable = observables[obs_id]
        if type_ == "observable":
            return observable.observable_placeholders[i]
        return observable.noise_placeholders[i]

    def quote(values) -> str:
        return ", ".join(f"`{v}'" for v in values)

    slots_to_transform = []
    errors = []
    for slot, values in _get_override_slots(problem).items():
        reparametrized = sorted(
            v.name
            for v in values
            if isinstance(v, sp.Symbol) and v.name in id_map
        )
        if not reparametrized:
            continue
        non_positive = sorted(
            float(v) for v in values if v.is_number and not v > 0
        )
        others = sorted(
            str(v)
            for v in values
            if not v.is_number
            and not (isinstance(v, sp.Symbol) and v.name in id_map)
        )
        if not non_positive and not others:
            slots_to_transform.append(slot)
            continue

        obs_id, type_, i = slot
        error = (
            f"* {type_.capitalize()} placeholder `{get_placeholder(*slot)}' "
            f"of observable `{obs_id}' (overridden by "
            f"{quote(reparametrized)}) also receives:"
        )
        if others:
            error += (
                f"\n  - parameters that are not reparametrized: "
                f"{quote(others)} (consider including them in "
                "`parameter_ids')"
            )
        if non_positive:
            error += "\n  - non-positive numbers: " + ", ".join(
                f"{v:g}" for v in non_positive
            )
        errors.append(error)

    if errors:
        raise NotImplementedError(
            "Cannot log-transform the following placeholders, which are "
            "overridden by parameters to be reparametrized. This requires "
            "all overrides of a placeholder to be reparametrized parameters "
            "or positive numbers.\n"
            + "\n".join(errors)
            + "\nSplitting observables is not supported. Exclude the "
            "affected parameters or split the observables manually."
        )

    if not slots_to_transform:
        return

    # placeholder -> b^placeholder in formulas
    for obs_id, type_, i in slots_to_transform:
        observable = observables[obs_id]
        placeholder = get_placeholder(obs_id, type_, i)
        if type_ == "observable":
            observable.formula = observable.formula.subs(
                placeholder, b_expr**placeholder
            )
        # observable placeholders may also occur in the noise formula
        observable.noise_formula = observable.noise_formula.subs(
            placeholder, b_expr**placeholder
        )

    # update override values
    def transform(value: sp.Basic) -> sp.Basic:
        if isinstance(value, sp.Symbol):
            return sp.Symbol(id_map[value.name], real=True)
        return sp.Float(log_b(float(value)))

    slots_to_transform = set(slots_to_transform)
    for m in problem.measurements:
        for type_, attr in (
            ("observable", "observable_parameters"),
            ("noise", "noise_parameters"),
        ):
            overrides = getattr(m, attr)
            if any(
                (m.observable_id, type_, i) in slots_to_transform
                for i in range(len(overrides))
            ):
                setattr(
                    m,
                    attr,
                    [
                        transform(v)
                        if (m.observable_id, type_, i) in slots_to_transform
                        else v
                        for i, v in enumerate(overrides)
                    ],
                )


def _add_sbml_log_rule(
    model: libsbml.Model,
    parameter_id: str,
    new_id: str,
    b_expr: sp.Expr,
    value: float | None,
) -> None:
    """Add parameter ``new_id`` and the rule ``parameter_id = b^new_id``."""
    add_sbml_parameter(model, new_id, value=value, constant=True)

    par = model.getParameter(parameter_id)
    check(par.setConstant(False))
    # an assignment rule and an initial assignment must not have the same
    #  target
    if model.getInitialAssignmentBySymbol(parameter_id) is not None:
        model.removeInitialAssignment(parameter_id)

    rule = model.createAssignmentRule()
    check(rule.setVariable(parameter_id))
    formula = f"exp({new_id})" if b_expr == sp.E else f"10^({new_id})"
    ast = libsbml.parseL3Formula(formula)
    if ast is None:
        raise RuntimeError(f"Failed to parse `{formula}'.")
    check(rule.setMath(ast))
