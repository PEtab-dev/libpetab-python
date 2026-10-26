"""Conversion of time-dependent piecewise assignments to experiments."""

from __future__ import annotations

import warnings
from copy import deepcopy

import libsbml
import sympy as sp
from sbmlmath import TimeSymbol, sbml_math_to_sympy

from .core import (
    Change,
    Condition,
    Experiment,
    ExperimentPeriod,
    Problem,
)
from .math import petab_symbol
from .models._sbml_utils import add_sbml_parameter
from .models.sbml_model import SbmlModel

__all__ = ["DEFAULT_EXPERIMENT_ID", "assignments_to_experiments"]


DEFAULT_EXPERIMENT_ID = "default_experiment"


def assignments_to_experiments(problem: Problem) -> Problem:
    """Replace time-dependent piecewise assignments by experiment periods.

    Fixed parameters may change over the course of a single simulation, e.g.,
    as described by a v2 experiment with multiple periods. As v1 did not have
    experiments, then these changes were
    sometimes directly encoded in an SBML model as assignment rules with
    piecewise functions. This method takes a v2 problem
    (possibly converted from a v1 problem) and replaces those SBML assignment
    rules with v2 experiments.

    .. note::

        Measurements without an experiment ID are assigned to a new experiment
        :data:`DEFAULT_EXPERIMENT_ID`, if needed, which starts at time 0.
        Every experiment keeps the period that starts it, without conditions
        if all of them were replaced, since a period's time overrides the
        model's initial time.

    **Scope:** conversion is supported for SBML assignment rules to
    parameters that depend on time through the conditions of piecewise
    functions, and at most linearly through their values, where

    * every such condition is an inequality that can be re-arranged
      explicitly into `t </<=/>/>= ...`,
    * every piecewise function that depends on time has an explicit
      "otherwise" piece,
    * every switching time evaluates to a real number in every experiment
      (it may depend on parameters that are fixed or set by the condition
      table, but not on estimated ones),
    * and all other symbols are truly constant model parameters.

    Within that, the piecewise functions may be combined: a sum of doses
    given at different times (`d1 * piecewise(0, t < t1, 1) + d2 *
    piecewise(0, t < t2, 1)`), phases one after another as several pieces or
    nested piecewise functions (`piecewise(b0, t < t1, piecewise(b1, t < t2,
    b2))`). Each switch that changes the value becomes a period. A value may
    also change linearly with time within a period (a ramp: `piecewise(r * t
    + a, t <= t_end, 0)`). A condition change is applied once at the start of
    its period, so the target of such an assignment becomes a differential
    entity instead, with the rate rule `d target/dt = <target>_slope`, and
    each period sets both the target and its slope; an assignment that only
    switches between constant values stays a constant parameter. Anything else
    raises a :class:`NotImplementedError`. Initial assignments stay in the
    model, since they are evaluated once, at the start of the simulation.

    Condition changes of the replaced assignments' symbols are dropped, unless
    the model, an observable or another change still refers to them.

    .. warning::

        As SBML assignment rules, the changes would occur during
        pre-equilibration. As v2 experiments, the changes do not occur during
        pre-equilibration. Hence, using this converter may affect the
        correctness of your model's pre-equilibration. In this case, you could
        e.g. keep the SBML assignment rules and/or add indicator variables to
        trigger them correctly in the model for some/all pre-equilibration
        conditions.


    :param problem: The PEtab problem to convert. This will not be modified.
    :returns: The converted PEtab problem.
    :raises NotImplementedError:
        If a time-dependent assignment is outside the scope described above.
    :raises ValueError:
        If the converted problem does not pass linting.
    """
    if not isinstance(problem.model, SbmlModel):
        raise TypeError("Only SBML models are supported.")

    problem = deepcopy(problem)
    sbml_model = problem.model.sbml_model

    # Initial assignments are evaluated once, at the start of the
    #  simulation, so they never switch and stay in the model.
    time_dependent = []
    for rule in sbml_model.getListOfRules():
        if not rule.isAssignment():
            continue
        expr = sbml_math_to_sympy(rule)
        if any(pw.has(TimeSymbol) for pw in expr.atoms(sp.Piecewise)):
            time_dependent.append((rule, expr))
    if not time_dependent:
        _check_lint(problem)
        return problem

    experiments = list(problem.experiments)
    if any(m.experiment_id is None for m in problem.measurements):
        default_experiment = Experiment(id=DEFAULT_EXPERIMENT_ID)
        experiments.append(default_experiment)
    else:
        default_experiment = None
    droppable = _add_start_periods(experiments)

    # (experiment, period start time, target ID, target value)
    additions = []
    replaced_symbols = set()
    converted_ids = []
    for rule, expr in time_dependent:
        assignment = _PiecewiseAssignment(rule, expr, problem)
        assignment.replace_in_model()
        for experiment in experiments:
            additions += [
                (experiment, *change)
                for change in assignment.changes(experiment)
            ]
        replaced_symbols |= assignment.symbol_ids
        converted_ids.append(assignment.target_id)

    if converted_ids and any(e.has_preequilibration for e in experiments):
        warnings.warn(
            "Switches of time-dependent piecewise assignments during "
            "preequilibration are not supported. The converted "
            f"assignments to {converted_ids} keep their value at time 0 "
            "throughout "
            "preequilibration.",
            stacklevel=2,
        )

    # replace the changes that are covered by the new conditions, unless
    #  something else still refers to their targets
    replaced_symbols -= _referenced_ids(problem, additions)
    _drop_changes(problem, experiments, replaced_symbols)
    _add_conditions(problem, additions)

    for experiment in experiments:
        experiment.sort_periods()
        experiment.periods = [
            p
            for p in experiment.periods
            if p.condition_ids or id(p) not in droppable
        ]

    if default_experiment and any(
        p.condition_ids for p in default_experiment.periods
    ):
        problem += default_experiment
        for measurement in problem.measurements:
            if measurement.experiment_id is None:
                measurement.experiment_id = DEFAULT_EXPERIMENT_ID

    _check_lint(problem)
    return problem


def _check_lint(problem: Problem) -> None:
    """Raise a :class:`ValueError` if the converted ``problem`` does not
    pass linting."""
    validation_issues = problem.validate()
    if validation_issues.has_errors():
        messages = "\n".join(f"{issue.message}" for issue in validation_issues)
        raise ValueError(
            f"The converted PEtab problem did not pass linting:\n{messages}"
        )


class _PiecewiseAssignment:
    """A time-dependent piecewise assignment to a model parameter, and the
    condition changes that replace it, see :func:`assignments_to_experiments`.
    """

    def __init__(
        self,
        rule: libsbml.Rule | libsbml.InitialAssignment,
        expr: sp.Expr,
        problem: Problem,
    ):
        """Check that the assignment ``rule``, with the value ``expr``, is
        supported."""
        self.rule = rule
        self.problem = problem
        self.sbml_model = problem.model.sbml_model
        self.target_id = rule.getId()
        if self.sbml_model.getParameter(self.target_id) is None:
            raise NotImplementedError(
                "Only assignments to parameters are supported, but "
                f"`{self.target_id}` is not a parameter."
            )
        self.expr = expr
        (self.time,) = expr.atoms(TimeSymbol)
        self.symbol_ids = {str(s) for s in expr.free_symbols}
        self.parameter_ids = {p.id for p in problem.parameters}
        self.switches = _supported_time_switches(
            expr,
            self.time,
            self.target_id,
            {p.id for p in problem.parameters if p.estimate},
        )
        if non_constant := sorted(
            str(s)
            for s in expr.free_symbols - {self.time}
            if not (p := self.sbml_model.getParameter(str(s)))
            or not p.getConstant()
        ):
            raise NotImplementedError(
                "Only constant model parameters are supported as symbols in "
                f"the assignment to `{self.target_id}`, but it refers to "
                f"{non_constant}."
            )
        # the value before every switch: a condition whose two sides grow
        #  apart with time holds only after its switch
        self.default = expr.xreplace(
            {
                c: sp.false if slope > 0 else sp.true
                for c, (_, slope) in self.switches.items()
            }
        )
        # unless a condition sets them, the switching times' symbols keep
        #  their value from the parameter table or else from the model
        self.nominal_values = {
            p.id: p.nominal_value for p in problem.parameters if not p.estimate
        }
        self.fixed_values = {
            s: self.nominal_values.get(str(s), self._model_value(s))
            for t_switch, _ in self.switches.values()
            for s in t_switch.free_symbols
        }
        # whether the value changes with time within a period
        self.ramps = expr.xreplace({c: sp.Dummy() for c in self.switches}).has(
            self.time
        )
        self.slope_id = None

    def replace_in_model(self) -> None:
        """Replace the assignment in the model by a target that the
        condition table sets.

        A value that changes with time within a period (a ramp) cannot be a
        condition change on a constant target, which is applied once at the
        start of its period. The model only makes the target a differential
        entity, `d target/dt = <target>_slope`; the slope's value, like the
        target's, is set by every period that starts one.
        """
        if self.ramps:
            self.slope_id = self.problem.get_unused_id(
                f"{self.target_id}_slope"
            )
            add_sbml_parameter(
                self.sbml_model, self.slope_id, value=0.0, constant=False
            )
            rate_rule = self.sbml_model.createRateRule()
            rate_rule.setVariable(self.target_id)
            rate_rule.setMath(libsbml.parseL3Formula(self.slope_id))
        model_values = {
            s: self._model_value(s)
            for s in self.default.free_symbols - {self.time}
        }
        target = self.sbml_model.getParameter(self.target_id)
        target.setValue(
            float(self.default.subs(self.time, 0).subs(model_values))
        )
        target.setConstant(False)
        self.rule.removeFromParentAndDelete()

    def changes(
        self, experiment: Experiment
    ) -> list[tuple[float, str, sp.Expr]]:
        """The changes that replace the assignment in ``experiment``, each as
        ``(period start time, target ID, target value)``."""
        changes = []
        periods = experiment.sorted_periods
        overrides = {}
        # the value at the end of the previous period
        held = None
        for i_period, period in enumerate(periods):
            period_changes = self.problem.get_changes_for_period(period)
            # a condition's changes are evaluated once, at the start of their
            #  period, all with the values from before the period
            overrides |= {
                c.target_id: self._at_period_start(c.target_value, overrides)
                for c in period_changes
            }
            subs = self.fixed_values | {
                s: overrides[str(s)]
                for s in self.expr.free_symbols
                if str(s) in overrides
            }
            if unknown := sorted(
                {
                    str(x)
                    for s in self.expr.free_symbols
                    if str(s) in overrides
                    for x in overrides[str(s)].free_symbols
                }
                - self.parameter_ids
            ):
                raise NotImplementedError(
                    "Only condition changes whose values are known at the "
                    "start of their period are supported, but in experiment "
                    f"`{experiment.id}`, the changes of the symbols of the "
                    f"assignment to `{self.target_id}` refer to {unknown}."
                )
            times = self._switching_times(subs, experiment.id)

            if period.is_preequilibration:
                # the value at time 0, which respects whether a condition
                #  holds for a switch at time 0, held there: a ramp does not
                #  run during preequilibration
                held = self.expr.subs(subs).subs(self.time, 0)
                changes.append(
                    (
                        period.time,
                        self.target_id,
                        self._with_model_values(held),
                    )
                )
                if self.ramps:
                    changes.append((period.time, self.slope_id, sp.S.Zero))
                continue

            start_value = self._value_from(subs, times, period.time)
            # each switch within the period that changes the value
            t_end = (
                periods[i_period + 1].time
                if i_period + 1 < len(periods)
                else sp.oo
            )
            switch_values = []
            previous = start_value
            for t in sorted({t for t in times if period.time < t < t_end}):
                value = self._value_from(subs, times, t)
                if not (value - previous).is_zero:
                    switch_values.append((t, value))
                previous = value

            if i_period == 0:
                # The model already has the pre-switch value at the start of
                #  the first period, which stays even without conditions,
                #  unless that value refers to parameter table symbols, for
                #  which the model only has a number. A ramp's start and
                #  slope are always set by the tables.
                if (
                    self.ramps
                    or not (start_value - self.default).is_zero
                    or {str(s) for s in self.default.free_symbols}
                    & self.parameter_ids
                ):
                    changes += self._start(
                        period.time, self._with_model_values(start_value)
                    )
            elif (
                {c.target_id for c in period_changes} & self.symbol_ids
                # preequilibration held the slope at zero
                or (self.ramps and periods[i_period - 1].is_preequilibration)
                # a switch at the start of the period
                or not (start_value - held).is_zero
            ):
                changes += self._start(period.time, start_value)
            for t, value in switch_values:
                changes += self._start(t, value)
            held = previous
        return changes

    def _model_value(self, symbol: sp.Symbol) -> float:
        """The value of the model parameter ``symbol``."""
        return self.sbml_model.getParameter(str(symbol)).getValue()

    def _at_period_start(self, value: sp.Expr, overrides: dict) -> sp.Expr:
        """The ``value`` of a change, evaluated at the start of its period.

        Symbols that a condition set before take that value (``overrides``),
        parameters fixed by the parameter table their nominal values, and the
        other constant model parameters outside the parameter table their
        model values. Estimated parameters stay, since no condition sets them;
        any other symbol stays as well, since its value is not known here.
        """
        if value is None:
            return value
        known = {}
        for s in value.free_symbols:
            if str(s) in overrides:
                known[s] = overrides[str(s)]
            elif str(s) in self.nominal_values:
                known[s] = sp.Float(self.nominal_values[str(s)])
            elif (
                str(s) not in self.parameter_ids
                and (p := self.sbml_model.getParameter(str(s))) is not None
                and p.getConstant()
            ):
                known[s] = sp.Float(p.getValue())
        return value.xreplace(known)

    def _with_model_values(self, value: sp.Expr) -> sp.Expr:
        """``value`` for a change at the start of an experiment, which may
        only refer to parameter table symbols: the other symbols are constant
        model parameters, at their values."""
        return value.subs(
            {
                s: self._model_value(s)
                for s in value.free_symbols - {self.time}
                if str(s) not in self.parameter_ids
            }
        )

    def _switching_times(self, subs: dict, experiment_id: str) -> list[float]:
        """The switching times, for the symbol values ``subs``."""
        times = []
        for t_switch, _ in self.switches.values():
            t = t_switch.subs(subs)
            if not (t.is_number and t.is_real and t.is_finite):
                raise NotImplementedError(
                    "Only switching times that evaluate to a real number are "
                    "supported, but a switching time of the assignment to "
                    f"`{self.target_id}` in experiment `{experiment_id}` is "
                    f"`{t}`."
                )
            times.append(float(t))
        return times

    def _value_from(self, subs: dict, times: list[float], t0: float):
        """The value from ``t0`` up to its next switch, for the symbol values
        ``subs``, every condition replaced by whether it holds in between."""
        later = [t for t in times if t > t0]
        probe = (t0 + min(later)) / 2 if later else t0 + 1
        truth = {c: c.subs(subs).subs(self.time, probe) for c in self.switches}
        return self.expr.xreplace(truth).subs(subs)

    def _start(
        self, t: float, value: sp.Expr
    ) -> list[tuple[float, str, sp.Expr]]:
        """The changes that start ``value`` at time ``t``: the target set to
        the value at ``t``, and, for a target that ramps, its slope set to
        the value's rate of change."""
        changes = [(t, self.target_id, value.subs(self.time, t))]
        if self.ramps:
            changes.append((t, self.slope_id, sp.diff(value, self.time)))
        return changes


def _add_start_periods(experiments: list[Experiment]) -> set[int]:
    """Give every experiment a period that starts it, and return the
    :func:`id` of the periods that are dropped if all their conditions are
    replaced.

    The period that starts an experiment stays, even without conditions: its
    time overrides the model's initial time, so the experiment would start at
    its first switch instead.
    """
    droppable = set()
    for experiment in experiments:
        if all(p.is_preequilibration for p in experiment.periods):
            experiment += ExperimentPeriod(time=0)
        start = min(
            (p for p in experiment.periods if not p.is_preequilibration),
            key=lambda p: p.time,
        )
        droppable |= {
            id(p)
            for p in experiment.periods
            if p.condition_ids and p is not start
        }
    return droppable


def _referenced_ids(
    problem: Problem,
    additions: list[tuple[Experiment, float, str, sp.Expr]],
) -> set[str]:
    """The IDs that the model, the observables, or the condition changes,
    including those in ``additions``, refer to."""
    ids = problem.model.get_referenced_ids()
    ids |= {
        str(s)
        for table in [*problem.observable_tables, *problem.condition_tables]
        for s in table.free_symbols
    }
    ids |= {str(s) for *_, value in additions for s in value.free_symbols}
    return ids


def _drop_changes(
    problem: Problem, experiments: list[Experiment], target_ids: set[str]
) -> None:
    """Drop the changes of ``target_ids`` from the condition tables, and the
    conditions left without changes from the periods of ``experiments``."""
    condition_ids = {c.id for c in problem.conditions}
    for table in problem.condition_tables:
        for condition in table.conditions:
            condition.changes = [
                c for c in condition.changes if c.target_id not in target_ids
            ]
        table.elements = [c for c in table.conditions if c.changes]
    removed_ids = condition_ids - {c.id for c in problem.conditions}
    for experiment in experiments:
        for period in experiment.periods:
            period.condition_ids = [
                cid for cid in period.condition_ids if cid not in removed_ids
            ]


def _add_conditions(
    problem: Problem,
    additions: list[tuple[Experiment, float, str, sp.Expr]],
) -> None:
    """For each ``(experiment, time, target ID, value)`` in ``additions``,
    add a condition with that change to the period of ``experiment`` that
    starts at ``time``, which is added if needed.

    Equal changes share a condition. Their values use the symbols that PEtab
    uses for the condition table, rather than those parsed from SBML.
    """
    new_conditions = {}
    for experiment, t, target_id, value in additions:
        value = value.xreplace(
            {
                s: petab_symbol(s.name)
                for s in value.free_symbols
                if not isinstance(s, TimeSymbol)
            }
        )
        if (cond_id := new_conditions.get((target_id, value))) is None:
            cond_id = problem.get_unused_id(f"cond_{len(new_conditions)}")
            new_conditions[target_id, value] = cond_id
            problem += Condition(
                id=cond_id,
                changes=[Change(target_id=target_id, target_value=value)],
            )
        period = next((p for p in experiment.periods if p.time == t), None)
        if period is None:
            experiment += (period := ExperimentPeriod(time=t))
        period.condition_ids.append(cond_id)


def _supported_time_switches(
    expr: sp.Expr, time: sp.Symbol, target_id: str, estimated_ids: set[str]
) -> dict[sp.Rel, tuple[sp.Expr, sp.Expr]]:
    """Check that ``expr`` depends on ``time`` only in the supported way, and
    return the conditions through which it does, each with the time at which
    it switches and the slope of its two sides' difference.

    ``expr`` may combine several piecewise functions, each with several
    pieces and nested in one another (a sum of doses, phases one after
    another), as long as time enters only through their conditions, each a
    single inequality linear in time, each piecewise function has an
    explicit "otherwise" piece, and no switching time depends on the
    parameters ``estimated_ids``. Anything else raises
    :class:`NotImplementedError`.
    """
    unsupported = NotImplementedError(
        "Only time dependence through piecewise conditions, each a single "
        "inequality linear in time, with an explicit otherwise piece, is "
        f"supported, but the assignment to `{target_id}` is `{expr}`."
    )
    switches = {}
    for pw in expr.atoms(sp.Piecewise):
        if not pw.has(time):
            continue
        # an explicit "otherwise" piece
        if pw.args[-1].cond != sp.true:
            raise unsupported
        for piece in pw.args[:-1]:
            condition = piece.cond
            if not condition.has(time):
                continue
            if not (
                # a supported inequality
                isinstance(condition, (sp.Gt, sp.Ge, sp.Lt, sp.Le))
                # that changes value at maximally a single time point
                and (
                    slope := sp.diff(condition.gts - condition.lts, time)
                ).is_number
                # and this time point is identifiable
                and slope.is_nonzero
            ):
                raise unsupported
            t_switch = sp.solve(condition.gts - condition.lts, time)[0]
            switches[condition] = (t_switch, slope)
    # outside the conditions, time enters at most linearly: the value may
    #  ramp, at a rate that does not itself change with time
    rest = expr.xreplace({c: sp.Dummy() for c in switches})
    if rest.has(time) and sp.diff(rest, time).has(time):
        raise NotImplementedError(
            "Only values that change at most linearly with time are "
            f"supported, but the assignment to `{target_id}` is `{expr}`."
        )
    # no period can start at an estimated time
    if estimated := sorted(
        {
            str(s)
            for t_switch, _ in switches.values()
            for s in t_switch.free_symbols
            if str(s) in estimated_ids
        }
    ):
        raise NotImplementedError(
            "Only switching times that are not estimated are supported, but "
            f"the switching times of the assignment to `{target_id}` depend "
            f"on the estimated parameters {estimated}."
        )
    return switches
