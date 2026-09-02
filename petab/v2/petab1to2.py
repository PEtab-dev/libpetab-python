"""Convert PEtab version 1 problems to version 2."""

from __future__ import annotations

import numbers
import re
import shutil
import warnings
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from uuid import uuid4

import libsbml
import pandas as pd
import sympy as sp
from pandas.io.common import get_handle, is_url
from sbmlmath import TimeSymbol, sbml_math_to_sympy

from .. import v1, v2
from ..v1.math import sympify_petab
from ..v1.yaml import get_path_prefix, load_yaml, validate
from ..versions import get_major_version
from .models import MODEL_TYPE_SBML
from .models.sbml_model import SbmlModel

__all__ = ["petab1to2"]


def petab1to2(
    yaml_config: Path | str,
    output_dir: Path | str | None = None,
    assignments_to_experiments: bool = False,
) -> v2.Problem | None:
    """Convert from PEtab 1.0 to PEtab 2.0 format.

    Convert a PEtab problem from PEtab 1.0 to PEtab 2.0 format.

    .. note::

       Some aspects of PEtab v1 were not well-defined. For example, model
       initialization order (e.g., applying initial assignments before or
       after condition table overrides) and the impact of compartment size
       changes were not specified. In such cases, we made assumptions that are
       consistent with the clarified PEtab v2 specifications,
       the PEtab test suite, or common practice.
       Therefore, it is recommended to carefully review the generated PEtab v2
       problem to ensure it aligns with the expected behavior.

    :param yaml_config:
        The PEtab problem as dictionary or YAML file name.
    :param output_dir:
        The output directory to save the converted PEtab problem, or ``None``,
        to return a :class:`petab.v2.Problem` instance.
    :param assignments_to_experiments:
        If ``True``, convert time-dependent piecewise assignments in the
        model to conditions and experiments in the tables.

    :raises ValueError:
        If the input is invalid or does not pass linting or if the generated
        files do not pass linting.
    """
    if output_dir is not None:
        return petab_files_1to2(
            yaml_config,
            output_dir,
            assignments_to_experiments=assignments_to_experiments,
        )

    with TemporaryDirectory() as tmp_dir:
        petab_files_1to2(
            yaml_config,
            tmp_dir,
            assignments_to_experiments=assignments_to_experiments,
        )
        return v2.Problem.from_yaml(Path(tmp_dir, Path(yaml_config).name))


def petab_files_1to2(
    yaml_config: Path | str | dict,
    output_dir: Path | str,
    assignments_to_experiments: bool = False,
):
    """Convert PEtab files from PEtab 1.0 to PEtab 2.0.


    :param yaml_config:
        The PEtab problem as dictionary or YAML file name.
    :param output_dir:
        The output directory to save the converted PEtab problem.
    :param assignments_to_experiments:
        If ``True``, convert time-dependent piecewise assignments in the
        model to conditions and experiments in the PEtab tables.

    :raises ValueError:
        If the input is invalid or does not pass linting or if the generated
        files do not pass linting.
    """
    if isinstance(yaml_config, Path | str):
        yaml_file = str(yaml_config)
        path_prefix = get_path_prefix(yaml_file)
        yaml_config = load_yaml(yaml_config)
        get_src_path = lambda filename: f"{path_prefix}/{filename}"  # noqa: E731
    else:
        yaml_file = None
        path_prefix = None
        get_src_path = lambda filename: filename  # noqa: E731

    get_dest_path = lambda filename: f"{output_dir}/{filename}"  # noqa: E731

    # Validate the original PEtab problem
    validate(yaml_config, path_prefix=path_prefix)
    if get_major_version(yaml_config) != 1:
        raise ValueError("PEtab problem is not version 1.")
    petab_problem = v1.Problem.from_yaml(yaml_file or yaml_config)
    # TODO: move to mapping table
    # get rid of conditionName column if present (unsupported in v2)
    petab_problem.condition_df = petab_problem.condition_df.drop(
        columns=[v1.C.CONDITION_NAME], errors="ignore"
    )
    if v1.lint_problem(petab_problem):
        raise ValueError("Provided PEtab problem does not pass linting.")

    # convert time-dependent piecewise assignments in the model to conditions
    #  and experiments; this modifies `petab_problem.model`
    piecewise_conditions, piecewise_periods = [], {}
    obsolete_columns, declared_values = [], {}
    if assignments_to_experiments:
        (
            piecewise_conditions,
            piecewise_periods,
            obsolete_columns,
            declared_values,
        ) = _assignments_to_experiments(
            petab_problem.model, petab_problem.condition_df
        )
        # the converted assignments made these condition table columns
        #  obsolete
        petab_problem.condition_df = petab_problem.condition_df.drop(
            columns=obsolete_columns
        )

    output_dir = Path(output_dir)

    # Update YAML file
    new_yaml_config = _update_yaml(yaml_config)
    new_yaml_config = v2.ProblemConfig(**new_yaml_config)

    # Update tables

    # parameter table
    parameter_df = v1v2_parameter_df(petab_problem.parameter_df.copy())
    v2.write_parameter_df(
        parameter_df, get_dest_path(new_yaml_config.parameter_files[0])
    )

    # copy files that don't need conversion: models
    model_files = [
        model.location for model in new_yaml_config.model_files.values()
    ]
    if piecewise_conditions:
        # the assignments were removed from the model, so we have to write
        #  the modified model instead of copying the original one
        if len(model_files) > 1:
            raise NotImplementedError(
                "Converting assignments to experiments is not supported for "
                "problems with multiple models."
            )
        libsbml.writeSBMLToFile(
            petab_problem.model.sbml_document, get_dest_path(model_files[0])
        )
    else:
        for file in model_files:
            _copy_file(get_src_path(file), Path(get_dest_path(file)))

    # Update observable table
    for observable_file in new_yaml_config.observable_files:
        observable_df = v1.get_observable_df(get_src_path(observable_file))
        observable_df = v1v2_observable_df(
            observable_df,
        )
        v2.write_observable_df(observable_df, get_dest_path(observable_file))

    # records for the experiment table to be created
    experiments = []
    # the change that each generated condition applies
    generated_changes = {
        record[v2.C.CONDITION_ID]: (
            record[v2.C.TARGET_ID],
            record[v2.C.TARGET_VALUE],
        )
        for record in piecewise_conditions
    }

    def apply_changes(
        condition_ids: list[str], values: dict, drop: bool = True
    ) -> list[str]:
        """Drop generated conditions that don't change anything.

        The targets of the converted assignments declare their pre-switch
        value in the model, so any change to that value is redundant until
        some period changes it. ``values`` tracks the value of each target
        during the experiment and is updated in place.
        """
        applied = []
        for condition_id in condition_ids:
            if (change := generated_changes.get(condition_id)) is None:
                # a condition from the v1 condition table
                applied.append(condition_id)
                continue
            target_id, target_value = change
            if drop and values.get(target_id) == target_value:
                continue
            values[target_id] = target_value
            applied.append(condition_id)
        return applied

    def condition_exists(condition_id: str) -> bool:
        """Check whether a condition will exist in the v2 condition table."""
        return bool(condition_id) and bool(
            petab_problem.condition_df.loc[condition_id].notna().any()
        )

    def create_experiment_id(sim_cond_id: str, preeq_cond_id: str) -> str:
        if not sim_cond_id and not preeq_cond_id:
            return ""
        # check whether the conditions will exist in the v2 condition table
        sim_cond_exists = condition_exists(sim_cond_id) or bool(
            piecewise_periods.get(sim_cond_id)
        )
        preeq_cond_exists = condition_exists(preeq_cond_id)
        if not sim_cond_exists and not preeq_cond_exists:
            # if we have only all-NaN conditions, we don't create a new
            #  experiment
            return ""

        if preeq_cond_id:
            preeq_cond_id = f"{preeq_cond_id}_"
        exp_id = f"experiment__{preeq_cond_id}__{sim_cond_id}"
        if exp_id in experiments:
            i = 1
            while f"{exp_id}_{i}" in experiments:
                i += 1
            exp_id = f"{exp_id}_{i}"
        return exp_id

    measured_experiments = (
        petab_problem.get_simulation_conditions_from_measurement_df()
    )
    # measurements that are not associated with any condition, and thus with
    #  any experiment
    unconditioned_measurements = False
    for (
        _,
        row,
    ) in measured_experiments.iterrows():
        # generate a new experiment for each simulation / pre-eq condition
        #  combination
        sim_cond_id = row[v1.C.SIMULATION_CONDITION_ID]
        preeq_cond_id = row.get(v1.C.PREEQUILIBRATION_CONDITION_ID, "")
        exp_id = create_experiment_id(sim_cond_id, preeq_cond_id)
        if not exp_id:
            unconditioned_measurements = True
            continue
        # the values that the converted targets have during this experiment
        values = dict(declared_values)
        if preeq_cond_id:
            preeq_condition_ids = (
                [preeq_cond_id] if condition_exists(preeq_cond_id) else []
            )
            # the converted assignments apply during preequilibration, too;
            #  since that is a steady state, only their first period applies
            if preeq_periods := piecewise_periods.get(preeq_cond_id):
                preeq_condition_ids += preeq_periods[0][1]
            experiments.extend(
                {
                    v2.C.EXPERIMENT_ID: exp_id,
                    v2.C.TIME: v2.C.TIME_PREEQUILIBRATION,
                    v2.C.CONDITION_ID: condition_id,
                }
                # the preequilibration period is required even if it does not
                #  change anything
                for condition_id in apply_changes(preeq_condition_ids, values)
                or [""]
            )
        if condition_exists(sim_cond_id):
            experiments.append(
                {
                    v2.C.EXPERIMENT_ID: exp_id,
                    v2.C.TIME: 0,
                    v2.C.CONDITION_ID: sim_cond_id,
                }
            )
        # the changes of the first period of an experiment must not refer to
        #  model symbols, so the redundant changes at its start are only
        #  dropped if some other period comes first, or if they are the only
        #  numeric ones
        cur_periods = piecewise_periods.get(sim_cond_id, [])
        keep_initial = (
            not preeq_cond_id
            and not condition_exists(sim_cond_id)
            and any(
                not isinstance(generated_changes[condition_id][1], float)
                for _, condition_ids in cur_periods
                for condition_id in condition_ids
            )
        )
        # the periods of the assignments converted for this condition
        for i, (time, condition_ids) in enumerate(cur_periods):
            experiments.extend(
                {
                    v2.C.EXPERIMENT_ID: exp_id,
                    v2.C.TIME: time,
                    v2.C.CONDITION_ID: condition_id,
                }
                for condition_id in apply_changes(
                    condition_ids, values, drop=not (keep_initial and i == 0)
                )
            )
    # measurements that are not associated with any condition still need an
    #  experiment to refer to the periods of the converted assignments
    default_experiment_id = ""
    if unconditioned_measurements and (
        default_periods := piecewise_periods.get("")
    ):
        taken_ids = {record[v2.C.EXPERIMENT_ID] for record in experiments} | (
            set(petab_problem.condition_df.index)
        )
        i = 0
        while (default_experiment_id := f"exp_{i}") in taken_ids:
            i += 1
        values = dict(declared_values)
        for time, condition_ids in default_periods:
            experiments.extend(
                {
                    v2.C.EXPERIMENT_ID: default_experiment_id,
                    v2.C.TIME: time,
                    v2.C.CONDITION_ID: condition_id,
                }
                for condition_id in apply_changes(condition_ids, values)
            )

    # experiments whose conditions all turned out to be redundant
    defined_experiments = {
        record[v2.C.EXPERIMENT_ID] for record in experiments
    }

    # Update condition table
    #  the generated conditions that no experiment ended up using are dropped,
    #  and the remaining ones are renumbered consecutively
    used_conditions = {record[v2.C.CONDITION_ID] for record in experiments}
    piecewise_conditions = [
        record
        for record in piecewise_conditions
        if record[v2.C.CONDITION_ID] in used_conditions
    ]
    taken_ids = set(petab_problem.condition_df.index)
    renamed_conditions = {}
    counter = 0
    for record in piecewise_conditions:
        while (new_id := f"cond_{counter}") in taken_ids:
            counter += 1
        counter += 1
        renamed_conditions[record[v2.C.CONDITION_ID]] = new_id
        record[v2.C.CONDITION_ID] = new_id
    for record in experiments:
        record[v2.C.CONDITION_ID] = renamed_conditions.get(
            record[v2.C.CONDITION_ID], record[v2.C.CONDITION_ID]
        )
    src_condition_files = list(new_yaml_config.condition_files)
    if piecewise_conditions and not src_condition_files:
        # there is no condition table yet to add the new conditions to
        new_yaml_config.condition_files.append("conditions.tsv")
        src_condition_files.append(None)
    for i, condition_file in enumerate(src_condition_files):
        condition_df = (
            v1v2_condition_df(
                v1.get_condition_df(get_src_path(condition_file)).drop(
                    columns=obsolete_columns, errors="ignore"
                ),
                petab_problem.model,
            )
            if condition_file is not None
            else pd.DataFrame(columns=v2.C.CONDITION_DF_REQUIRED_COLS)
        )
        if i == 0 and piecewise_conditions:
            # the conditions generated from the model assignments go to the
            #  first condition table
            new_rows = pd.DataFrame(piecewise_conditions)
            condition_df = (
                pd.concat([condition_df, new_rows], ignore_index=True)
                if not condition_df.empty
                else new_rows
            )
        v2.write_condition_df(
            condition_df, get_dest_path(new_yaml_config.condition_files[i])
        )

    if experiments:
        exp_table_path = output_dir / "experiments.tsv"
        if exp_table_path.exists():
            raise ValueError(
                f"Experiment table file {exp_table_path} already exists."
            )
        new_yaml_config.experiment_files.append("experiments.tsv")
        v2.write_experiment_df(
            v2.get_experiment_df(pd.DataFrame(experiments)), exp_table_path
        )

    for measurement_file in new_yaml_config.measurement_files:
        measurement_df = v1.get_measurement_df(get_src_path(measurement_file))
        # if there is already an experiment ID column, we rename it
        if v2.C.EXPERIMENT_ID in measurement_df.columns:
            measurement_df.rename(
                columns={v2.C.EXPERIMENT_ID: f"experiment_id_{uuid4()}"},
                inplace=True,
            )
        # add pre-eq condition id if not present or convert to string
        #  for simplicity
        if v1.C.PREEQUILIBRATION_CONDITION_ID in measurement_df.columns:
            measurement_df.fillna(
                {v1.C.PREEQUILIBRATION_CONDITION_ID: ""}, inplace=True
            )
        else:
            measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID] = ""

        if (
            petab_problem.condition_df is not None
            and len(
                set(petab_problem.condition_df.columns) - {v1.C.CONDITION_NAME}
            )
            == 0
            and not any(piecewise_periods.values())
        ):
            # we can't have "empty" conditions with no overrides in v2,
            #  therefore, we drop the respective condition ID completely
            #   TODO: or can we?
            # TODO: this needs to be checked condition-wise, not globally
            measurement_df[v1.C.SIMULATION_CONDITION_ID] = ""
            if v1.C.PREEQUILIBRATION_CONDITION_ID in measurement_df.columns:
                measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID] = ""
        # condition IDs to experiment IDs
        measurement_df.insert(
            0,
            v2.C.EXPERIMENT_ID,
            measurement_df.apply(
                lambda row: create_experiment_id(
                    row[v1.C.SIMULATION_CONDITION_ID],
                    row.get(v1.C.PREEQUILIBRATION_CONDITION_ID, ""),
                ),
                axis=1,
            ),
        )
        # experiments without any change were not created
        measurement_df.loc[
            ~measurement_df[v2.C.EXPERIMENT_ID].isin(defined_experiments),
            v2.C.EXPERIMENT_ID,
        ] = ""
        if default_experiment_id:
            # measurements without condition are assigned to the experiment
            #  that only contains the converted model assignments
            measurement_df[v2.C.EXPERIMENT_ID] = measurement_df[
                v2.C.EXPERIMENT_ID
            ].replace("", default_experiment_id)
        del measurement_df[v1.C.SIMULATION_CONDITION_ID]
        del measurement_df[v1.C.PREEQUILIBRATION_CONDITION_ID]
        v2.write_measurement_df(
            measurement_df, get_dest_path(measurement_file)
        )

    # Write the new YAML file
    new_yaml_file = output_dir / Path(yaml_file).name
    new_yaml_config.to_yaml(new_yaml_file)

    # validate updated Problem
    validation_issues = v2.lint_problem(new_yaml_file)

    if validation_issues:
        sev = v2.lint.ValidationIssueSeverity
        validation_issues.log(max_level=sev.WARNING)
        errors = "\n".join(
            map(
                str,
                (i for i in validation_issues if i.level > sev.WARNING),
            )
        )
        if errors:
            raise ValueError(
                "The generated PEtab v2 problem did not pass linting: "
                f"{errors}"
            )


def _time_symbol(expr: sp.Expr) -> TimeSymbol | None:
    """Get the model time symbol occurring in the given expression."""
    return next(
        (sym for sym in expr.free_symbols if isinstance(sym, TimeSymbol)), None
    )


def _time_threshold(condition: sp.Basic) -> tuple[str, float] | None:
    """Solve the condition of some piecewise branch for model time.

    :param condition: The condition to solve.
    :returns:
        The comparison operator (with model time on the left-hand side) and
        the time at which the branch changes, or ``None`` if the condition is
        not a comparison that can be solved for a fixed point in time. The
        time may be symbolic, e.g. some parameter that the condition table
        sets.
    """
    if not isinstance(condition, sp.core.relational.Relational):
        # e.g. a conjunction of several comparisons
        return None
    if (op := condition.rel_op) not in ("<", "<=", ">", ">="):
        return None
    if (time := _time_symbol(condition)) is None:
        return None

    # `lhs <op> rhs` is equivalent to `lhs - rhs <op> 0`, which can be solved
    #  for time if it is linear in time
    try:
        poly = sp.Poly(condition.lhs - condition.rhs, time)
    except sp.PolynomialError:
        return None
    if poly.degree() != 1:
        return None

    slope = poly.coeff_monomial(time)
    if not slope.is_number:
        return None
    if slope < 0:
        # dividing by a negative slope mirrors the comparison
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]

    return op, -poly.coeff_monomial(1) / slope


def _evaluate_comparison(time: float, op: str, threshold: float) -> bool:
    """Evaluate a comparison as returned by :func:`_time_threshold`.

    ``time`` is interpreted as the start of a half-open period, i.e., the
    comparison is evaluated for a time infinitesimally larger than ``time``.
    Thus, ``<`` and ``<=`` (and, respectively, ``>`` and ``>=``) cannot be
    distinguished -- the value of a single point in time is irrelevant here.
    """
    match op:
        case "<" | "<=":
            return time < threshold
        case ">" | ">=":
            return time >= threshold
    raise AssertionError(f"Unknown operator {op}.")


def _resolve_piecewise(expr: sp.Expr, time: float) -> sp.Expr:
    """Evaluate all piecewise sub-expressions at the given time.

    Only to be used for expressions that passed the checks in
    :func:`_piecewise_to_periods`, which ensure that a branch can be selected
    for any point in time.
    """

    def select_branch(piecewise: sp.Piecewise) -> sp.Expr:
        for value, condition in piecewise.args:
            if condition is sp.true:
                return value
            threshold = _time_threshold(condition)
            if (
                threshold is not None
                and threshold[1].is_number
                and _evaluate_comparison(time, threshold[0], threshold[1])
            ):
                return value
        raise AssertionError(f"No branch of {piecewise} applies at {time}.")

    return expr.replace(lambda e: isinstance(e, sp.Piecewise), select_branch)


def _piecewise_to_periods(expr: sp.Expr) -> list[tuple[float, sp.Expr]] | None:
    """Split a time-dependent expression into periods of constant value.

    The piecewise expressions may occur anywhere inside ``expr``, e.g. in
    ``some_parameter * piecewise(0, time < 10, 1)``.

    :param expr: The right-hand side of some assignment.
    :returns:
        A list of ``(start time, value)`` tuples, ordered by start time, or
        ``None`` if the value of ``expr`` is not piecewise constant in time.
        An expression without any piecewise sub-expression is constant, and
        thus yields a single period -- substituting condition-specific values
        may well collapse a piecewise expression to a constant.
    """
    piecewises = expr.atoms(sp.Piecewise)
    if not piecewises:
        return None if _time_symbol(expr) is not None else [(0.0, expr)]

    breakpoints = set()
    for piecewise in piecewises:
        *branches, (_, fallback_condition) = piecewise.args
        if fallback_condition is not sp.true:
            # without an `otherwise`, the value is undefined outside the
            #  given intervals
            return None
        for _, condition in branches:
            threshold = _time_threshold(condition)
            if threshold is None or not threshold[1].is_number:
                # not a piecewise expression that switches at some fixed,
                #  known point in time
                return None
            breakpoints.add(float(threshold[1]))

    if not breakpoints:
        return None
    breakpoints = sorted(breakpoints)

    # the breakpoints split the time axis into segments of constant value
    periods = []
    for i, start in enumerate([float("-inf"), *breakpoints]):
        # a point inside the current segment at which we evaluate the
        #  conditions; for the leading segment, any point before the first
        #  breakpoint will do
        probe = breakpoints[0] - 1 if i == 0 else start
        value = _resolve_piecewise(expr, probe)
        if _time_symbol(value) is not None:
            # the remaining time-dependence cannot be expressed by conditions
            return None
        # anything before t=0 is covered by the first period
        start = max(start, 0.0)

        if periods and periods[-1][0] == start:
            # segments before t=0 are collapsed into a single period
            periods[-1] = (start, value)
        elif periods and periods[-1][1] == value:
            # no change compared to the previous period
            continue
        else:
            periods.append((start, value))

    return periods


def _pre_switch_value(expr: sp.Expr) -> sp.Expr | None:
    """Get the value of an expression before all of its switching times.

    As time approaches minus infinity, ``time < x`` holds and ``time > x``
    does not -- whatever the threshold ``x`` is. The value before the first
    switch can therefore be determined even for switching times that are only
    known per condition.

    :returns:
        The value, or ``None`` if it cannot be determined.
    """
    for piecewise in expr.atoms(sp.Piecewise):
        selected = None
        for value, condition in piecewise.args:
            if condition is sp.true:
                selected = value
                break
            threshold = _time_threshold(condition)
            if threshold is None:
                return None
            if threshold[0] in ("<", "<="):
                selected = value
                break
        if selected is None:
            return None
        expr = expr.subs(piecewise, selected)

    if expr.atoms(sp.Piecewise) or _time_symbol(expr) is not None:
        return None
    return expr


def _set_declared_value(element: libsbml.SBase, value: float) -> bool:
    """Set the value that some model entity declares.

    :returns: Whether the value could be set.
    """
    if isinstance(element, libsbml.Parameter):
        element.setValue(value)
    elif isinstance(element, libsbml.Compartment):
        element.setSize(value)
    elif isinstance(element, libsbml.Species):
        if element.isSetInitialConcentration():
            element.setInitialConcentration(value)
        elif element.isSetInitialAmount():
            element.setInitialAmount(value)
        else:
            return False
    else:
        return False
    return True


def _referenced_symbols(sbml_model: libsbml.Model) -> set[str]:
    """Get the IDs of all model entities that are referenced by some math."""

    def ast_names(node: libsbml.ASTNode) -> Iterator[str]:
        if node is None:
            return
        if node.getType() == libsbml.AST_NAME:
            yield node.getName()
        for i in range(node.getNumChildren()):
            yield from ast_names(node.getChild(i))

    symbols = set()
    for element in sbml_model.getListOfAllElements():
        if getattr(element, "isSetMath", bool)():
            symbols |= set(ast_names(element.getMath()))
        # the target of some assignment counts as being used, too
        for getter in ("getVariable", "getSymbol"):
            if hasattr(element, getter):
                symbols.add(getattr(element, getter)())
    return symbols


def _condition_value(value: float | str) -> sp.Expr:
    """Convert a value from the PEtab v1 condition table to a sympy expr."""
    if isinstance(value, numbers.Number):
        return sp.Float(value)
    return sympify_petab(value)


def _assignments_to_experiments(
    model: v1.Model, condition_df: pd.DataFrame | None
) -> tuple[
    list[dict],
    dict[str, list[tuple[float, list[str]]]],
    list[str],
    dict[str, float],
]:
    """Turn time-dependent piecewise assignments into conditions/periods.

    Any (initial) assignment whose value is piecewise constant in time is
    removed from ``model`` and is replaced by one condition per time interval.
    The value that applied before the first switch is written to the target's
    declared value in the model, so that the conditions only have to encode
    the actual changes.

    The switching times and the values may be given by parameters that the
    PEtab v1 condition table sets. Those are resolved for each condition
    separately, so that the resulting periods generally differ between
    conditions.

    :param model: The model to convert. Modified in place.
    :param condition_df: The PEtab v1 condition table.
    :returns:
        Records for the condition table, the experiment periods for each v1
        condition as ``(start time, condition IDs)`` tuples ordered by start
        time, the condition table columns that the conversion made obsolete,
        and the values that the converted targets now declare in the model.
    """
    if not isinstance(model, SbmlModel):
        raise NotImplementedError(
            "Converting assignments to experiments is only supported for "
            f"SBML models, but got {type(model).__name__}."
        )
    sbml_model = model.sbml_model

    # the values to substitute for each v1 condition; if there are no
    #  conditions, the assignments are resolved with the model values alone
    if condition_df is not None and len(condition_df):
        substitutions = {
            condition_id: row.dropna().to_dict()
            for condition_id, row in condition_df.iterrows()
        }
    else:
        substitutions = {"": {}}

    taken_ids = set(substitutions)
    conditions = []
    # condition IDs by the change they apply, for re-using conditions
    condition_ids = {}
    # the conditions to apply at any given start time, by v1 condition
    periods = {condition_id: {} for condition_id in substitutions}
    converted = []
    counter = 0

    assignments = [
        *(rule for rule in sbml_model.getListOfRules() if rule.isAssignment()),
        *sbml_model.getListOfInitialAssignments(),
    ]
    for assignment in assignments:
        expr = sbml_math_to_sympy(assignment)
        if not expr.atoms(sp.Piecewise):
            # nothing to convert
            continue
        target_id = (
            assignment.getVariable()
            if isinstance(assignment, libsbml.Rule)
            else assignment.getSymbol()
        )

        # resolve the assignment for every condition; if that is not possible
        #  for any single one of them, the assignment is left in the model
        resolved = {}
        for condition_id, values in substitutions.items():
            subs = {
                symbol: _condition_value(values[symbol.name])
                for symbol in expr.free_symbols
                if symbol.name in values
            }
            cur_periods = _piecewise_to_periods(
                expr.subs(subs) if subs else expr
            )
            if cur_periods is None:
                break
            resolved[condition_id] = cur_periods
        else:
            first_values = set()
            for condition_id, cur_periods in resolved.items():
                for i, (time, value) in enumerate(cur_periods):
                    target_value = (
                        float(value) if value.is_Number else str(value)
                    )
                    if i == 0:
                        first_values.add(target_value)
                    change = (target_id, target_value)
                    if change not in condition_ids:
                        while (new_id := f"cond_{counter}") in taken_ids:
                            counter += 1
                        taken_ids.add(new_id)
                        counter += 1
                        condition_ids[change] = new_id
                        conditions.append(
                            {
                                v2.C.CONDITION_ID: new_id,
                                v2.C.TARGET_ID: target_id,
                                v2.C.TARGET_VALUE: target_value,
                            }
                        )
                    periods[condition_id].setdefault(time, []).append(
                        condition_ids[change]
                    )
            # the value before the first switch is the value the target
            #  declares in the model; if the switching times leave no room
            #  for it, fall back to the value all conditions start with
            declared_value = _pre_switch_value(expr)
            if declared_value is None or not declared_value.is_Number:
                declared_value = (
                    first_values.pop() if len(first_values) == 1 else None
                )
            else:
                declared_value = float(declared_value)
            converted.append((assignment, target_id, expr, declared_value))

    declared_values = {}
    for assignment, target_id, _, declared_value in converted:
        if isinstance(assignment, libsbml.Rule):
            sbml_model.removeRuleByVariable(target_id)
        else:
            sbml_model.removeInitialAssignment(target_id)
        # the assignment is now handled by the condition table, but the target
        #  still needs a value in the model
        target = sbml_model.getElementBySId(target_id)
        if isinstance(declared_value, float) and _set_declared_value(
            target, declared_value
        ):
            declared_values[target_id] = declared_value

    # condition table columns that only served the converted assignments are
    #  now obsolete -- their effect is covered by the generated conditions
    referenced = _referenced_symbols(sbml_model)
    resolved_symbols = {
        symbol.name
        for _, _, expr, _ in converted
        for symbol in expr.free_symbols
    } - {target_id for _, target_id, _, _ in converted}
    obsolete_columns = [
        column
        for column in (
            condition_df.columns if condition_df is not None else []
        )
        if column in resolved_symbols and column not in referenced
    ]

    return (
        conditions,
        {
            condition_id: sorted(times.items())
            for condition_id, times in periods.items()
        },
        obsolete_columns,
        declared_values,
    )


def _update_yaml(yaml_config: dict) -> dict:
    """Update PEtab 1.0 YAML to PEtab 2.0 format."""
    yaml_config = yaml_config.copy()

    # Update format_version
    yaml_config[v2.C.FORMAT_VERSION] = "2.0.0"

    # Add extensions
    yaml_config[v2.C.EXTENSIONS] = {}

    # Move models and set IDs (filename for now)
    yaml_config[v2.C.MODEL_FILES] = {}
    for problem in yaml_config[v1.C.PROBLEMS]:
        models = {}
        for sbml_file in problem[v1.C.SBML_FILES]:
            model_id = sbml_file.split("/")[-1].split(".")[0]
            models[model_id] = {
                v2.C.MODEL_LANGUAGE: MODEL_TYPE_SBML,
                v2.C.MODEL_LOCATION: sbml_file,
            }
            yaml_config[v2.C.MODEL_FILES] |= models
        del problem[v1.C.SBML_FILES]

        for file_type in (
            v1.C.CONDITION_FILES,
            v1.C.MEASUREMENT_FILES,
            v1.C.OBSERVABLE_FILES,
        ):
            if file_type in problem:
                yaml_config[file_type] = problem[file_type]
                del problem[file_type]
    del yaml_config[v1.C.PROBLEMS]

    # parameter_file -> parameter_files
    if not isinstance(
        (par_files := yaml_config.pop(v1.C.PARAMETER_FILE, [])), list
    ):
        par_files = [par_files]
    yaml_config[v2.C.PARAMETER_FILES] = par_files

    return yaml_config


def _copy_file(src: Path | str, dest: Path):
    """Copy file."""
    # src might be a URL - convert to Path if local
    src_url = urlparse(src)
    if not src_url.scheme:
        src = Path(src)
    elif src_url.scheme == "file" and not src_url.netloc:
        src = Path(src.removeprefix("file:/"))

    if is_url(src):
        with (
            get_handle(src, mode="r") as src_handle,
            open(dest, "w") as dest_handle,
        ):
            dest_handle.write(src_handle.handle.read())
        return

    try:
        if dest.samefile(src):
            return
    except FileNotFoundError:
        shutil.copy(str(src), str(dest))


def v1v2_condition_df(
    condition_df: pd.DataFrame, model: v1.Model
) -> pd.DataFrame:
    """Convert condition table from petab v1 to v2."""
    condition_df = condition_df.copy().reset_index()
    with suppress(KeyError):
        # conditionName was dropped in PEtab v2
        condition_df.drop(columns=[v1.C.CONDITION_NAME], inplace=True)

    condition_df = condition_df.melt(
        id_vars=[v1.C.CONDITION_ID],
        var_name=v2.C.TARGET_ID,
        value_name=v2.C.TARGET_VALUE,
    ).dropna(subset=[v2.C.TARGET_VALUE])

    if condition_df.empty:
        # This happens if there weren't any condition-specific changes
        return pd.DataFrame(
            columns=[
                v2.C.CONDITION_ID,
                v2.C.TARGET_ID,
                v2.C.TARGET_VALUE,
            ]
        )

    return condition_df


def v1v2_observable_df(observable_df: pd.DataFrame) -> pd.DataFrame:
    """Convert observable table from petab v1 to v2.

    Perform all updates that can be done solely on the observable table:
    * drop observableTransformation, update noiseDistribution
    * update placeholder parameters
    """
    df = observable_df.copy().reset_index()

    # drop observableTransformation, update noiseDistribution
    #  if there is no observableTransformation, no need to update
    if v1.C.OBSERVABLE_TRANSFORMATION in df.columns:
        df[v1.C.OBSERVABLE_TRANSFORMATION] = df[
            v1.C.OBSERVABLE_TRANSFORMATION
        ].fillna(v1.C.LIN)

        if v1.C.NOISE_DISTRIBUTION in df:
            df[v1.C.NOISE_DISTRIBUTION] = df[v1.C.NOISE_DISTRIBUTION].fillna(
                v1.C.NORMAL
            )
        else:
            df[v1.C.NOISE_DISTRIBUTION] = v1.C.NORMAL

        # merge observableTransformation into noiseDistribution
        def get_noise_dist(row):
            dist = row.get(v1.C.NOISE_DISTRIBUTION)
            trans = row.get(v1.C.OBSERVABLE_TRANSFORMATION)

            if trans == v1.C.LIN:
                new_dist = dist
            else:
                new_dist = f"{trans}-{dist}"

            if new_dist == "log10-normal":
                warnings.warn(
                    f"Noise distribution `{new_dist}' for "
                    f"observable `{row[v1.C.OBSERVABLE_ID]}'"
                    f" is not supported in PEtab v2. "
                    "Using `log-normal` instead.",
                    # call to `petab1to2`
                    stacklevel=9,
                )
                new_dist = v2.C.LOG_NORMAL

            if new_dist not in v2.C.NOISE_DISTRIBUTIONS:
                raise NotImplementedError(
                    f"Noise distribution `{new_dist}' for "
                    f"observable `{row[v1.C.OBSERVABLE_ID]}'"
                    f" is not supported in PEtab v2."
                )

            return new_dist

        df[v2.C.NOISE_DISTRIBUTION] = df.apply(get_noise_dist, axis=1)
        df.drop(columns=[v1.C.OBSERVABLE_TRANSFORMATION], inplace=True)

    def extract_placeholders(row: pd.Series, type_: str) -> str:
        """Extract placeholders from observable formula."""
        if type_ == "observable":
            formula = row[v1.C.OBSERVABLE_FORMULA]
        elif type_ == "noise":
            formula = row[v1.C.NOISE_FORMULA]
        else:
            raise ValueError(f"Unknown placeholder type: {type_}")

        if pd.isna(formula):
            return ""

        t = f"{re.escape(type_)}Parameter"
        o = re.escape(row[v1.C.OBSERVABLE_ID])

        pattern = re.compile(rf"(?:^|\W)({t}\d+_{o})(?=\W|$)")

        expr = sympify_petab(formula)
        # for 10+ placeholders, the current lexicographical sorting will result
        #  in incorrect ordering of the placeholder IDs, so that they don't
        #  align with the overrides in the measurement table, but who does
        #  that anyway?
        return v2.C.PARAMETER_SEPARATOR.join(
            sorted(
                str(sym)
                for sym in expr.free_symbols
                if sym.is_Symbol and pattern.match(str(sym))
            )
        )

    df[v2.C.OBSERVABLE_PLACEHOLDERS] = df.apply(
        extract_placeholders, args=("observable",), axis=1
    )
    df[v2.C.NOISE_PLACEHOLDERS] = df.apply(
        extract_placeholders, args=("noise",), axis=1
    )

    return df


def v1v2_parameter_df(
    parameter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert parameter table from petab v1 to v2.

    Do all the necessary conversions to the parameter table that can
    be done with the parameter table alone.
    """
    df = parameter_df.copy().reset_index()

    # parameter.estimate: int -> bool
    df[v2.C.ESTIMATE] = df[v1.C.ESTIMATE].apply(
        lambda x: str(bool(int(x))).lower()
    )

    def update_prior(row):
        """Convert prior to v2 format."""
        prior_type = row.get(v1.C.OBJECTIVE_PRIOR_TYPE)
        if pd.isna(prior_type):
            prior_type = v1.C.UNIFORM

        pscale = row.get(v1.C.PARAMETER_SCALE)
        if pd.isna(pscale):
            pscale = v1.C.LIN

        if prior_type not in v1.C.PARAMETER_SCALE_PRIOR_TYPES:
            return prior_type

        new_prior_type = prior_type.removeprefix("parameterScale").lower()
        if pscale != v1.C.LIN:
            new_prior_type = f"{pscale}-{new_prior_type}"

        if new_prior_type == "log10-normal":
            warnings.warn(
                f"Prior distribution `{new_prior_type}' for parameter "
                f"`{row[v1.C.PARAMETER_ID]}' is not supported in PEtab v2. "
                "Using `log-normal` instead.",
                # call to `petab1to2`
                stacklevel=9,
            )
            new_prior_type = v2.C.LOG_NORMAL

        if new_prior_type not in v2.C.PRIOR_DISTRIBUTIONS:
            raise NotImplementedError(
                f"PEtab v2 does not support prior type `{new_prior_type}' "
                f"required for parameter `{row[v1.C.PARAMETER_ID]}'."
            )

        return new_prior_type

    # update parameterScale*-priors
    if v1.C.OBJECTIVE_PRIOR_TYPE in df.columns:
        df[v1.C.OBJECTIVE_PRIOR_TYPE] = df.apply(update_prior, axis=1)

    # rename objectivePrior* to prior*
    df.rename(
        columns={
            v1.C.OBJECTIVE_PRIOR_TYPE: v2.C.PRIOR_DISTRIBUTION,
            v1.C.OBJECTIVE_PRIOR_PARAMETERS: v2.C.PRIOR_PARAMETERS,
        },
        inplace=True,
        errors="ignore",
    )
    # some columns were dropped in PEtab v2
    if v1.C.INITIALIZATION_PRIOR_TYPE in df and (
        df[v1.C.INITIALIZATION_PRIOR_TYPE].notna().any()
    ):
        warnings.warn(
            "Initialisation priors in parameter table are not supported "
            "in PEtab v2.",
            stacklevel=9,
        )
    if not (df[v1.C.PARAMETER_SCALE] == v1.C.LIN).all():
        warnings.warn(
            "Parameter scales are not supported in PEtab v2.",
            stacklevel=9,
        )
    df.drop(
        columns=[
            v1.C.INITIALIZATION_PRIOR_TYPE,
            v1.C.INITIALIZATION_PRIOR_PARAMETERS,
            v1.C.PARAMETER_SCALE,
        ],
        inplace=True,
        errors="ignore",
    )

    # if uniform, we need to explicitly set the parameters
    def update_prior_pars(row):
        prior_type = row.get(v2.C.PRIOR_DISTRIBUTION)
        prior_pars = row.get(v2.C.PRIOR_PARAMETERS)

        if prior_type in (v2.C.UNIFORM, v2.C.LOG_UNIFORM) and pd.isna(
            prior_pars
        ):
            return (
                f"{row[v2.C.LOWER_BOUND]}{v2.C.PARAMETER_SEPARATOR}"
                f"{row[v2.C.UPPER_BOUND]}"
            )

        return prior_pars

    df[v2.C.PRIOR_PARAMETERS] = df.apply(update_prior_pars, axis=1)

    return df
