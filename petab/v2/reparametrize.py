"""Reparametrization of PEtab problems."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Literal

import libsbml
import sympy as sp

from . import C
from .core import Parameter, PriorDistribution, Problem
from .math import petab_symbol
from .models._sbml_utils import add_sbml_parameter, set_initial_assignment
from .models.sbml_model import SbmlModel

__all__ = ["reparametrize_log"]


def _log10(x: float) -> float:
    return math.log10(x) if x != 0 else -math.inf


def _log(x: float) -> float:
    return math.log(x) if x != 0 else -math.inf


#: function(prior parameters of x, log_b, ln(b))
#:  -> prior parameters of log_b(x)
_PriorParameterConverter = Callable[
    [list[float], Callable[[float], float], float], list[float]
]

#: Priors of a parameter x that can be expressed as priors of log_b(x),
#:  mapped to (prior type of log_b(x), prior parameter converter)
_LOG_PRIOR_TO_THETA: dict[str, tuple[str, _PriorParameterConverter]] = {
    C.LOG_NORMAL: (
        C.NORMAL,
        lambda pars, log_function, log_factor: [p / log_factor for p in pars],
    ),
    C.LOG_LAPLACE: (
        C.LAPLACE,
        lambda pars, log_function, log_factor: [p / log_factor for p in pars],
    ),
    C.LOG_UNIFORM: (
        C.UNIFORM,
        lambda pars, log_function, log_factor: [log_function(p) for p in pars],
    ),
}


def reparametrize_log(
    problem: Problem,
    parameter_ids: Iterable[str],
    *,
    base: Literal[10, "e"] = 10,
    priors: dict[str, tuple[str, list[float]] | None] | None = None,
) -> tuple[Problem, dict[str, str]]:
    """Estimate parameters on log scale.

    For each parameter ``p``, replace ``p`` in the parameter table by a new
    parameter ``q = log_base(p)`` and define ``p = base^q``.
    This is the PEtab v2 way of estimating parameters on log scale
    (and of specifying priors for the log-transformed parameter).

    The given problem is assumed to be valid and is not modified.
    In the returned copy:

    * Parameter table: ``p`` is replaced by ``q`` with log-transformed bounds
      and nominal value. A lower bound of ``0`` becomes ``-inf``.
      Bounds of non-estimated parameters are dropped.
      A nominal value of ``0`` becomes ``-inf``.
      Estimated parameters without a non-negative nominal value get no
      nominal value for ``q``.
    * Model: If ``p`` is a model parameter, ``q`` is added to the model,
      and ``p`` is defined by the initial assignment ``p = base^q``,
      replacing any existing initial assignment for ``p``.
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
        The PEtab problem.
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
        Priors of non-estimated parameters are not converted and have to be
        dropped explicitly via ``{p: None}``.
    :return:
        The reparametrized problem and a dictionary mapping old to new
        parameter IDs.
    :raises ValueError:
        If a parameter cannot be reparametrized, e.g., because it is estimated
        and its lower bound is negative, if its prior cannot be
        expressed in terms of the new parameter, or if the new parameter
        would have an infinite bound and no prior, while other estimated
        parameters have priors.
    :raises NotImplementedError:
        If the problem has features that are not supported yet,
        e.g., multiple or non-SBML models, required extensions, mapping
        table entries for the parameters, priors of non-estimated parameters,
        or placeholders with overrides that cannot be log-transformed.
    """
    parameter_ids = list(dict.fromkeys(parameter_ids))
    priors = priors or {}
    if base == 10:
        log_factor = math.log(10)
        # not `_log(x) / log_factor`, which is inexact for powers of 10
        log_function = _log10
        b_expr = sp.Integer(10)
        prefix = "log10_"
    elif base == "e":
        log_factor = 1.0
        log_function = _log
        b_expr = sp.E
        prefix = "log_"
    else:
        raise ValueError(f"Unsupported base: {base}")

    if unknown := set(priors) - set(parameter_ids):
        raise ValueError(f"Priors given for unknown parameters: {unknown}")

    problem = deepcopy(problem)
    if not parameter_ids:
        return problem, {}

    if problem.extensions.sciml is not None:
        raise NotImplementedError(
            "Reparametrization of PEtab SciML problems is not supported."
        )
    if problem.config and (
        required := [
            ext_id
            for ext_id, ext in problem.config.extensions.items()
            if ext.required
        ]
    ):
        raise NotImplementedError(
            "Reparametrization of problems with required extensions "
            f"({', '.join(required)}) is not supported."
        )

    if len(problem.models) > 1:
        raise NotImplementedError(
            "Reparametrization of multi-model problems is not supported."
        )
    model = problem.model

    new_parameters = {}
    model_entities = []
    for p in parameter_ids:
        if p not in problem.get_x_ids():
            raise ValueError(f"Parameter `{p}' is not in the parameter table.")
        par = problem[p]
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
            sbml_model = model.sbml_model
            if (sbml_model.getLevel(), sbml_model.getVersion()) < (2, 2):
                raise NotImplementedError(
                    "Reparametrization of model parameters requires "
                    "initial assignments (SBML L2V2 or later)."
                )
            model_entities.append(p)
        if par.estimate and (par.lb < 0 or par.ub <= 0):
            # (bounds are required for estimated parameters)
            raise ValueError(
                f"Parameter `{p}' must have a non-negative lower bound and "
                f"a positive upper bound, but got [{par.lb}, {par.ub}]."
            )

        if par.nominal_value is None or par.nominal_value < 0:
            if not par.estimate:
                raise ValueError(
                    f"Non-estimated parameter `{p}' must have a non-negative "
                    f"nominal value, but got {par.nominal_value}."
                )
            warnings.warn(
                f"Parameter `{p}' has no non-negative nominal value "
                f"({par.nominal_value}). The nominal value of its "
                "reparametrization is left empty.",
                stacklevel=2,
            )
            nominal_value = None
        else:
            nominal_value = log_function(par.nominal_value)

        if p in priors and priors[p] is not None:
            prior_distribution, prior_parameters = priors[p]
            prior_distribution = PriorDistribution(prior_distribution)
        elif p in priors or par.prior_distribution is None:
            prior_distribution, prior_parameters = None, []
        elif not par.estimate:
            raise NotImplementedError(
                f"Non-estimated parameter `{p}' has a prior. "
                "Reparametrization of priors of non-estimated parameters "
                "is not supported. "
                f"Use `priors={{'{p}': None}}` to drop it."
            )
        elif par.prior_distribution in _LOG_PRIOR_TO_THETA:
            prior_distribution, f = _LOG_PRIOR_TO_THETA[par.prior_distribution]
            prior_parameters = f(
                par.prior_parameters, log_function, log_factor
            )
        else:
            raise ValueError(
                f"Prior `{par.prior_distribution}' of parameter `{p}' "
                "cannot be expressed in terms of the log-transformed "
                "parameter."
            )

        if par.estimate:
            lb = log_function(par.lb)
            ub = log_function(par.ub)
        else:
            # bounds of non-estimated parameters are irrelevant
            lb = ub = None
        new_parameters[p] = Parameter(
            id=p,
            name=par.name and f"{prefix.removesuffix('_')}({par.name})",
            lb=lb,
            ub=ub,
            nominal_value=nominal_value,
            estimate=par.estimate,
            prior_distribution=prior_distribution,
            prior_parameters=prior_parameters,
            **(par.model_extra or {}),
        )

    # If any estimated parameter has a prior, estimated parameters without
    #  a prior default to uniform(lb, ub), which is improper for infinite
    #  bounds. Priors of non-estimated parameters are ignored.
    improper = [
        p
        for p, par in new_parameters.items()
        if par.estimate
        and par.prior_distribution is None
        and (math.isinf(par.lb) or math.isinf(par.ub))
    ]
    if improper and any(
        par.estimate
        and new_parameters.get(par.id, par).prior_distribution is not None
        for par in problem.parameters
    ):
        raise ValueError(
            f"Parameters {improper} would have infinite bounds and no "
            "prior, while other parameters have priors. Specify proper "
            "priors via `priors`."
        )

    # assign new IDs
    id_map = {}
    for p in parameter_ids:
        id_map[p] = new_parameters[p].id = problem.get_unused_id(
            f"{prefix}{p}", exclude=id_map.values()
        )

    _reparametrize_overrides(problem, id_map, b_expr, log_function)

    # substitute p -> b^q in all expressions
    def subs(expr: sp.Basic | None) -> sp.Basic | None:
        if expr is None:
            return expr
        return expr.subs(
            {
                s: b_expr ** petab_symbol(id_map[s.name])
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
        _add_sbml_log_assignment(
            model.sbml_model,
            p,
            id_map[p],
            b_expr,
            new_parameters[p].nominal_value,
        )

    return problem, id_map


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
    log_function,
) -> None:
    """Log-transform placeholders overridden by reparametrized parameters.

    A placeholder ``ph`` is substituted by ``b^ph`` if any of its overrides
    is a reparametrized parameter. This requires all its overrides
    (across all measurements of the observable) to be reparametrized
    parameters or positive numbers. Otherwise, the observable would have to
    be split, which is not supported.
    """
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
                f"{quote(others)}"
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
            + "\nSplitting observables is not supported. Reparametrize "
            "all or none of the parameters overriding a placeholder, "
            "or split the observables manually."
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
            return petab_symbol(id_map[value.name])
        return sp.Float(log_function(float(value)))

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


def _add_sbml_log_assignment(
    model: libsbml.Model,
    parameter_id: str,
    new_id: str,
    b_expr: sp.Expr,
    value: float | None,
) -> None:
    """Add parameter ``new_id`` and the initial assignment
    ``parameter_id = b^new_id``."""
    add_sbml_parameter(model, new_id, value=value, constant=True)
    # any existing initial assignment is overridden by the parameter table
    #  value anyway
    with warnings.catch_warnings():
        # `b^q` is printed the same for all SBML levels
        warnings.filterwarnings(
            "ignore", "SBML level < 3 has not been thoroughly tested"
        )
        set_initial_assignment(
            model,
            parameter_id,
            b_expr ** petab_symbol(new_id),
            literals_dimensionless=False,
        )
