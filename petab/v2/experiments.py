"""Functions operating on the PEtab timecouse table."""
from __future__ import annotations
import operator

import copy
#from dataclasses import dataclass, field
from pydantic.dataclasses import dataclass
import math
from pathlib import Path
from typing import TypeVar, TYPE_CHECKING

import pandas as pd
from graphlib import TopologicalSorter
from more_itertools import one

from . import core, lint, measurements
from .. import v1
from .C import (
    #EXPERIMENT,
    #EXPERIMENT_DELIMITER,
    EXPERIMENT_ID,
    #EXPERIMENT_NAME,
    INPUT_ID,
    INPUT_VALUE,
    PRIORITY,
    REPEAT_EVERY,
    #PERIOD_DELIMITER,
    TIME,
    TIME_STEADY_STATE,
)

if TYPE_CHECKING:
    from . import Problem

__all__ = [
    "get_experiment_df",
    "write_experiment_df",
    "Experiment",
    "Period",
]


Time = TypeVar("Time", float, int, str)
UpdateValue = TypeVar("UpdateValue", float, int, str)


def get_experiment_df(experiment_file: str | Path | None) -> pd.DataFrame:
    """Read the provided experiment file into a ``pandas.Dataframe``.

    Arguments:
        experiment_file:
            Location of PEtab experiment file, or a ``pandas.Dataframe``.

    Returns:
        The experiments dataframe.
    """
    if experiment_file is None:
        return experiment_file

    if isinstance(experiment_file, (str, Path)):
        experiment_file = pd.read_csv(
            experiment_file,
            sep="\t",
            float_precision="round_trip",
        )

    lint.assert_no_leading_trailing_whitespace(
        experiment_file.columns.values, EXPERIMENT
    )

    if not isinstance(experiment_file.index, pd.RangeIndex):
        experiment_file.reset_index(inplace=True)

    try:
        experiment_file.set_index([EXPERIMENT_ID], inplace=True)
    except KeyError as e:
        raise KeyError(
            f"Experiment table missing mandatory field {EXPERIMENT_ID}."
        ) from e

    return experiment_file


def write_experiment_df(df: pd.DataFrame, filename: str | Path) -> None:
    """Write the provided PEtab experiment table to disk.

    Arguments:
        df:
            The PEtab experiment table.
        filename:
            The table will be written to this location.
    """
    df.to_csv(filename, sep="\t", index=True)


#@dataclass
#class Period:
#    """A experiment period.
#
#    Attributes:
#        condition_id:
#            The PEtab condition ID.
#        end_time:
#            The time when the period ends.
#        start_time:
#            The time when the period starts.
#        experiment_id:
#            The ID of the experiment that this period belongs to.
#        last_period:
#            Whether this period is the last in the experiment.
#    """
#
#    condition_id: str
#    end_time: Time
#    start_time: Time
#    experiment_id: str
#    last_period: bool
#
#    def get_condition(self, condition_df: pd.DataFrame) -> pd.Series:
#        return condition_df.loc[self.condition_id]
#
#    def get_measurements(
#        self,
#        measurement_df: pd.DataFrame,
#    ) -> pd.Series:
#        experiment_measurement_df = measurements.get_experiment_measurements(
#            measurement_df=measurement_df, experiment_id=self.experiment_id
#        )
#
#        after_start = experiment_measurement_df[TIME] >= self.start_time
#        before_end = experiment_measurement_df[TIME] < self.end_time
#        if self.last_period:
#            # include measurements at the "end" of the period
#            # ...probably useless since this should always be `inf`.
#            before_end |= experiment_measurement_df[TIME] == self.end_time
#            # include steady-state measurements
#            before_end |= experiment_measurement_df[TIME].apply(
#                lambda time: core.time_is_at_steady_state(
#                    time=time,
#                    postequilibration=True,
#                )
#            )
#        return experiment_measurement_df.loc[after_start & before_end]


@dataclass
class Period:
    """An experiment period.

    Attributes:
        experiment_id:
            The experiment ID.
        input_id:
            The input ID.
        input_value:
            The input value.
        time:
            The time
            TODO rename to start_time
        priority:
            The order in which simultaneous periods are applied.
        repeat_every:
            The periodicity of the period.
    """
    experiment_id: str
    input_id: str
    input_value: float | str | None = None
    time: float = 0
    priority: int = 0
    repeat_every: float | None = None  # if float, then positive

    def denest(self, experiments: Experiments, time0: float, time1: float) -> list[Period]:
        """Denest a period.

        Args:
            experiments:
                The experiments that will be used to denest this period.
            time0:
                The `time` of this period.
            time1:
                The `time` of the next period, in the experiment that this period belongs to.

        Returns:
            The denested period, as a list of periods that do not reference other experiments in the `inputId`.
        """
        periods = []


        # Period is not nested -- simply return itself
        if (experiment := experiments.get(self.input_id, None)) is None:
            periods.append(copy.deepcopy(self))
            return periods

        # Shift start time of nested experiment to start time of this period
        experiment = experiment.to_new_start_time(start_time=time0)

        repeat_time1 = time1
        if self.repeat_every:
            if core.time_is_at_steady_state(time1):
                raise ValueError(
                    f"Unable to denest a repeating period with non-finite duration. Period: `{self}`"
                )
            n_repeats = math.ceil((time1-time0)/self.repeat_every)
            # If repeating, only denest up to a single repeat for now
            repeat_time1 = time0 + self.repeat_every
            if time1 < repeat_time1:
                repeat_time1 = time1

        # Add all nested periods until. `time0` accumulates the duration of
        # each added period.
        for nested_period_index, nested_period in enumerate(experiment.periods):
            # The period ends when its duration is over, or the repeat ends.
            period_time1 = min(repeat_time1, time0 + experiment.get_period_duration(nested_period_index))
            periods.extend(nested_period.denest(
                experiments=experiments,
                time0=time0,
                time1=period_time1,
            ))

            time0 += experiment.get_period_duration(period_index=nested_period_index)

            if time0 > repeat_time1:
                break

        if not self.repeat_every:
            return periods

        single_experiment = Experiment(experiment_id=self.experiment_id, periods=periods)
        repeated_experiments = [single_experiment.to_new_start_time(start_time=self.time + n_repeat * self.repeat_every) for n_repeat in range(n_repeats)]
        periods_before_time1 = [
            period
            for repeated_experiment in repeated_experiments
            for period in repeated_experiment.periods
            if period.time < time1
        ]
        return periods_before_time1


class Experiment:
    """An experiment.

    Attributes:
        experiment_id:
            The experiment ID.
        periods:
            A list of :class:`Period`.
    """

    def __init__(self, experiment_id: str, periods: list[Period]):
        # Sort by priority then time, to ensure periods are applied in the correct order and at the correct time.
        self.periods = sorted(
            periods,
            key=operator.attrgetter(PRIORITY, TIME),
        )
        self.set_experiment_id(experiment_id=experiment_id)

    def set_experiment_id(self, experiment_id: str):
        self.experiment_id = experiment_id
        for period in self.periods:
            period.experiment_id = self.experiment_id

    @staticmethod
    def from_periods(experiment_id: str, periods: list[Period]):
        return Experiment(experiment_id=experiment_id, periods=periods)

    @property
    def t0(self) -> float:
        return self.periods[0].time

    def get_period_end_time(self, period_index: int) -> float:
        """Get the end time of a period.

        Args:
            period_index:
                The index of the period.

        Returns:
            The end time of the period.
        """
        try:
            next_period = self.periods[period_index+1]
            end_time = next_period.time
        except IndexError:
            end_time = TIME_STEADY_STATE
        return end_time

    def get_period_duration(self, period_index: int) -> float:
        """Get the duration of a period.

        Args:
            period_index:
                The index of the period.

        Returns:
            The duration of the period.
        """
        return self.get_period_end_time(period_index=period_index) - self.periods[period_index].time

    def to_new_start_time(self, start_time: float) -> Experiment:
        """Get an experiment that has a new start time.

        All period start times are shifted such that the durations of each period don't change.

        Args:
            start_time:
                The new start time.

        Returns:
            The experiment, with a new start time.
        """
        shift = start_time - self.periods[0].time
        shifted_periods = copy.deepcopy(self.periods)
        for shifted_period in shifted_periods:
            shifted_period.time += shift
        return Experiment(experiment_id=self.experiment_id, periods=shifted_periods)

    def denest(self, experiments: Experiments) -> Experiment:
        denested_periods = [
            denested_period
            for period_index, period in enumerate(self.periods)
            for denested_period in period.denest(
                experiments=experiments,
                time0=period.time,
                time1=self.get_period_end_time(period_index=period_index),
            )
        ]
        return Experiment(experiment_id=self.experiment_id, periods=denested_periods)


class Experiments:
    """A set of experiments.

    Attributes:
        experiments:
            The experiments as a `dict` where keys are experiment IDs, and values are the corresponding :class:`Experiment`.
    """

    def __init__(self, experiments: list[Experiment]):
        self.experiments = {experiment.experiment_id: experiment for experiment in experiments}

    def get(self, key: str, default: Any):
        if key not in self:
            return default
        return self[key]

    def __getitem__(self, key: str):
        return self.experiments[key]

    def __setitem__(self, key: str, value: Experiment):
        self.experiments[key] = value

    def __delitem__(self, key: str):
        del self.experiments[key]

    def __iter__(self):
        # TODO
        pass

    def __len__(self):
        return len(self.experiments)

    def __contains__(self, key: str):
        return key in self.experiments

    def denest(self, experiment_id):
        return self.experiments[experiment_id].denest(experiments=self)

    @staticmethod
    def from_periods(periods: list[Period]):
        # Group periods by experiment
        experiment_periods = {}
        for period in periods:
            if period.experiment_id not in experiment_periods:
                experiment_periods[period.experiment_id] = []
            experiment_periods[period.experiment_id].append(period)

        experiments = [
            Experiment.from_periods(experiment_id=experiment_id, periods=periods)
            for experiment_id, periods in experiment_periods.items()
        ]
        return Experiments(experiments=experiments)

    @staticmethod
    def load(table: str | Path | pd.DataFrame) -> Experiments:
        if isinstance(table, str | Path):
            table = pd.read_csv(table, sep="\t")

        table = table.replace({float("nan"): None})

        periods = [Period(**{k: v for k, v in row.items() if v is not None}) for _, row in table.iterrows()]
        return Experiments.from_periods(periods=periods)

    def save(self, filename: str | Path = None) -> pd.DataFrame:
        """Get the table, and optionally save it to disk.

        Args:
            filename:
                The location where the table will be saved to disk.

        Returns:
            The table.
        """
        table = pd.DataFrame(data=[period for experiment in self.experiments.values() for period in experiment.periods])
        if filename is not None:
            table.to_csv(filename, sep="\t", index=False)
        return table


def get_v1_condition(periods: Period | list[Periods]) -> pd.DataFrame:
    """Convert (simultaneous) periods into a petab.v1 condition table.

    Args:
        periods:
            The simultaneous periods, or a single period.

    Returns:
        The condition table.
    """
    if not isinstance(periods, list):
        periods = [periods]

    if len(set(period.experiment_id for period in periods)) != 1:
        raise ValueError("All periods must belong to the same experiment.")

    condition_df = v1.get_condition_df(pd.DataFrame(data={
        v1.CONDITION_ID: [periods[0].experiment_id],
        **{
            period.input_id: [period.input_value]
            for period in periods
        },
    }))
    return condition_df


def get_period_measurements(period_index: int, experiment: Experiment, measurement_df: pd.DataFrame) -> pd.DataFrame:
    """Get the measurements associated with a specific experiment period.

    Args:
        period_index:
            The index of the period.
        experiment:
            The experiment.
        measurement_df:
            The table with measurements for (potentially all) periods and other
            experiments.

    Returns:
        The measurement table corresponding to the period.
    """
    experiment_measurement_df = measurement_df.loc[measurement_df[EXPERIMENT_ID] == experiment.experiment_id]
    period_measurement_df = experiment_measurement_df.loc[
        (experiment_measurement_df[TIME] >= experiment.periods[period_index].time)
        &
        (experiment_measurement_df[TIME] < experiment.get_period_end_time(period_index))
    ]
    return period_measurement_df

#  dict[float, v1.Problem]:

def get_v1_tables_sequence(experiments: Experiments, measurement_df: pd.DataFrame, experiment_ids: list[str] = None) -> dict[str, tuple[list[float], tuple[pd.DataFrame, pd.DataFrame]]]:
    """Convert experiments into a sequences of petab.v1 tables.

    This enables simulation of complicated experiments, simply by simulating
    a sequence of petab.v1 problems then joining the simulation together.

    Some petab.v2 features cannot be converted and will raise a `TypeError`.

    TODO support preequilibration.

    Args:
        experiments:
            The experiments.
        measurement_df:
            The measurements.

    Returns:
        Keys are experiment IDs (which are the condition IDs in the petab.v1
        tables). Values are:
        #. A list of start times for each experiment period.
        #. The (1) condition and (2) measurement table corresponding to each
           period.
    """
    if experiment_ids is None:
        experiment_ids = measurement_df[EXPERIMENT_ID].unique()
    result = {}

    for experiment_id in experiment_ids:
        print('create tables..', end=" ")
        print(experiment_id)
        experiment = experiments.denest(experiment_id)

        v1_tables_sequence = {}
        times = []
        active_periods = {}
        for period_index, period in enumerate(experiment.periods):
            # All quantity changes defined in previous periods, which haven't
            # been updated by this period, should remain "active".
            active_periods[period.input_id] = period

            period_measurement_df = get_period_measurements(period_index=period_index, experiment=experiment, measurement_df=measurement_df)
            period_measurement_df = period_measurement_df.rename(columns={EXPERIMENT_ID: v1.SIMULATION_CONDITION_ID})

            if period.time not in v1_tables_sequence:
                v1_tables_sequence[period.time] = [[], []]

            v1_tables_sequence[period.time][0].extend(active_periods.values())
            v1_tables_sequence[period.time][1].append(period_measurement_df)


        for time, (periods, measurement_tables) in v1_tables_sequence.items():
            v1_tables_sequence[time] = (get_v1_condition(periods), pd.concat(measurement_tables))

        result[experiment_id] = v1_tables_sequence
    return result


def get_v1_problem_sequence(petab_problem: Problem, experiment_ids: list[str] = None) -> dict[str, tuple[list[float], v1.Problem]]:
    """Convert a v2 problem into a one sequence of v1 problems per experiment.

    With this, tools can simulate the model of a v2 problem, as a sequence of
    models provided in the v1 problem format.

    TODO support preequilibration?

    Args:
        petab_problem:
            The v2 problem.

    Returns:
        Keys are experiment IDs (which are the condition IDs in the petab.v1
        tables). Values are:
        #. A list of start times for each experiment period.
        #. The petab problem corresponding to each experiment period.
    """
    v1_tables_sequence = get_v1_tables_sequence(experiments=petab_problem.experiments, measurement_df=petab_problem.measurement_df, experiment_ids=experiment_ids)

    result = {}
    for experiment_id, v1_table_data in v1_tables_sequence.items():
        print('create problem..', end=" ")
        print(experiment_id)
        v1_problems = []
        for time, (condition_df, measurement_df) in v1_table_data.items():
            #v1_problem = copy.deepcopy(petab_problem)
            v1_problem = v1.Problem(
                condition_df=condition_df,
                measurement_df=measurement_df,
                model=copy.deepcopy(petab_problem.model),
                observable_df=copy.deepcopy(petab_problem.observable_df),
                parameter_df=copy.deepcopy(petab_problem.parameter_df),
                mapping_df=copy.deepcopy(petab_problem.mapping_df),
            )

            #v1_problem.condition_df = condition_df
            #v1_problem.measurement_df = measurement_df
            v1_problems.append(v1_problem)
        result[experiment_id] = dict(zip(v1_table_data.keys(), v1_problems, strict=True))
    return result

#@dataclass
#class Experiment:
#    """A experiment.
#
#
#    Attributes:
#        name:
#            The experiment name.
#        periods:
#            The periods of the experiment.
#        #t0:
#        #    The time when the experiment starts.
#        experiment_id:
#            The experiment ID.
#    """
#
#    experiment_id: str
#    periods: list[Period]
#    name: None | str = None
#
#    @property
#    def timepoints(self):
#        _timepoints = [self.t0]
#        # skip last since it has indefinite duration
#        for period in self.periods[:-1]:
#            _timepoints.append(_timepoints[-1] + period.duration)
#        return _timepoints
#
#    @property
#    def condition_ids(self):
#        return [period.condition_id for period in self.periods]
#
#    def to_series(self):
#        period_definitions = []
#        for period in self.periods:
#            period_definition = PERIOD_DELIMITER.join(
#                [
#                    str(period.start_time),
#                    period.condition_id,
#                ]
#            )
#            period_definitions.append(period_definition)
#        experiment_definition = EXPERIMENT_DELIMITER.join(period_definitions)
#
#        series = pd.Series(
#            data={EXPERIMENT: experiment_definition},
#            name=self.experiment_id,
#        )
#        return series
#
#    @staticmethod
#    def from_df_row(
#        row: pd.Series,
#        # measurement_df: pd.DataFrame = None,
#        experiments: dict[str, Experiment] = None,
#    ) -> Experiment:
#        """Create a experiment object from a row definition.
#
#        Any nested or repetitive structure is flattened.
#
#        Argments:
#            row:
#                The row definition.
#            experiments:
#                Required to flatten nested experiments. Keys are experiment
#                IDs.
#
#        Returns:
#            The experiment.
#        """
#        experiment_id = row.name
#
#        # Collect all periods, which are possibly nested in one of two ways:
#        #    1. `restartEvery` is specified
#        #    2. another experimentId is used as the condition for a period
#        # We use "true period" to refer to a period as it is seen in a
#        # experiment table, i.e. as truth as specified by the user. We use
#        # "denested period" to refer to one of the possibly-many periods that
#        # actually occur within the true period, after it is denested.
#        true_periods = []
#        for period_definition in row.get(EXPERIMENT).split(
#            EXPERIMENT_DELIMITER
#        ):
#            true_periods.append(_parse_period_definition(period_definition))
#
#        # Denest any actually nested periods
#        denested_periods = []
#        for true_period_index, true_period in enumerate(true_periods):
#            (
#                true_start_time,
#                restart_every,
#                true_period_condition_id,
#            ) = true_period
#
#            # Determine the end time of the current period.
#            # The period ends when the next period starts, or never.
#            try:
#                true_end_time = true_periods[true_period_index + 1][0]
#            except IndexError:
#                true_end_time = TIME_STEADY_STATE
#
#            # TODO for now, require fixed time points for proper denesting
#            try:
#                true_start_time = float(true_start_time)
#                true_end_time = float(true_end_time)
#            except ValueError as e:
#                raise ValueError(
#                    "Parameterized experiment times are not yet supported. "
#                    f"In experiment {experiment_id}, there is a period "
#                    f"`{true_period_index}` starts at `{true_start_time}` and "
#                    f"ends at `{true_end_time}`."
#                ) from e
#
#            # TODO for now, require finite period durations if `restartEvery`
#            # nesting is specified
#            if restart_every and (
#                core.time_is_at_steady_state(
#                    true_start_time, preequilibration=True
#                )
#                or core.time_is_at_steady_state(
#                    true_end_time, postequilibration=True
#                )
#            ):
#                raise ValueError(
#                    "Period restarts are currently not supported in "
#                    "periods that are simulated until steady-state. This "
#                    f"occurs in experiment `{experiment_id}` during period "
#                    f"`{true_period_index}`. Period start time: "
#                    f"`{true_start_time}`. Period end time: `{true_end_time}`."
#                )
#
#            # Create lists of start and end times for each period repeat
#            # due to `restartEvery` nesting
#            restart_end_times = []
#            # Add initial period start
#            restart_start_times = [true_start_time]
#            if restart_every:
#                next_restart = true_start_time
#                while (
#                    next_restart := next_restart + restart_every
#                ) < true_end_time:
#                    # Add end of previous period restart
#                    restart_end_times.append(next_restart)
#                    # Add start of next period restart
#                    restart_start_times.append(next_restart)
#            # Add final period end
#            restart_end_times.append(true_end_time)
#
#            # Generate a new period for each restart due to `restartEvery`
#            # nesting
#            for restart_start_time, restart_end_time in zip(
#                restart_start_times, restart_end_times, strict=True
#            ):
#                denested_period = [
#                    Period(
#                        start_time=restart_start_time,
#                        end_time=restart_end_time,
#                        condition_id=true_period_condition_id,
#                        experiment_id=experiment_id,
#                        last_period=core.time_is_at_steady_state(
#                            restart_end_time,
#                            postequilibration=True,
#                        ),
#                    )
#                ]
#
#                # Handle `experimentId` condition nesting
#                if nested_experiment := experiments.get(
#                    true_period_condition_id
#                ):
#                    denested_period = copy.deepcopy(nested_experiment.periods)
#
#                    # TODO for now, require this
#                    if denested_period[0].start_time != 0:
#                        raise ValueError(
#                            "The first period of a nested experiment must "
#                            "have a start time of `0`. The experiment "
#                            f"`{nested_experiment.experiment_id}` is nested "
#                            f"within experiment `{experiment_id}`."
#                        )
#
#                    # Shift all nested periods in time, to match the start time
#                    # of the period restart. Drop nested periods that start
#                    # after the current period restart ends.
#                    current_start_time = restart_start_time
#                    for nested_experiment_period in denested_period:
#                        nested_experiment_period.experiment_id = experiment_id
#
#                        nested_experiment_period.start_time = (
#                            current_start_time
#                        )
#                        nested_experiment_period.end_time += current_start_time
#
#                        if (
#                            nested_experiment_period.end_time
#                            > restart_end_time
#                        ):
#                            nested_experiment_period.end_time = (
#                                restart_end_time
#                            )
#                            # Drop nested periods if they will start after the
#                            # current restarted period is scheduled to end.
#                            break
#
#                        current_start_time = nested_experiment_period.end_time
#
#                denested_periods.extend(denested_period)
#
#        return Experiment(
#            experiment_id=experiment_id,
#            name=row.get(EXPERIMENT_NAME),
#            periods=denested_periods,
#        )
#
#    @staticmethod
#    def from_df(
#        experiment_df: pd.DataFrame,
#        experiment_id: str | list[str] = None,
#    ) -> dict[str, Experiment] | Experiment:
#        """Read in all experiment(s).
#
#        Arguments:
#            experiment_df:
#                The PEtab experiment table.
#            experiment_ids:
#                Specific experiment id(s).
#
#        Returns:
#            The experiment(s).
#        """
#        experiment_df = _clean_experiment_df(experiment_df)
#        # TODO performance gain: only return experiments required to fully
#        # denest the supplied `experiment_ids` experiments. Currently returns
#        # all experiments in an order that supports denesting.
#        sorted_experiment_ids = toposort_experiments(experiment_df)
#
#        if experiment_id is None:
#            experiment_ids = sorted_experiment_ids
#        if isinstance(experiment_id, str):
#            experiment_ids = [experiment_id]
#
#        experiments = {}
#        for experiment_id, row in experiment_df.loc[
#            sorted_experiment_ids
#        ].iterrows():
#            experiments[experiment_id] = Experiment.from_df_row(
#                row=row,
#                experiments=experiments,
#            )
#
#        experiments = {
#            experiment_id: experiment
#            for experiment_id, experiment in experiments.items()
#            if experiment_id in experiment_ids
#        }
#
#        if len(experiment_ids) == 1:
#            experiments = experiments[one(experiment_ids)]
#
#        return experiments
#
#    def __len__(self):
#        return len(self.periods)


def _clean_experiment_df(experiment_df: pd.DataFrame) -> str:
    experiment_df.loc[:, EXPERIMENT] = experiment_df.loc[:, EXPERIMENT].apply(
        lambda experiment_definition: "".join(experiment_definition.split())
    )
    return experiment_df


def _parse_period_definition(period_def: str) -> list[float | str]:
    """Parse a period definition.

    The expected definition format is:

        startTime[:restartEvery]:conditionId

    Arguments:
        period_def:
            The period definition, in the same format as specified in the
            experiment table.

    Returns:
        A tuple with:
            1. The period `startTime`.
            2. The period `restartEvery`.
            3. The period condition ID.
    """
    data = period_def.split(PERIOD_DELIMITER)

    restart_every = 0
    if len(data) == 2:
        start_time, condition_id = data
    elif len(data) == 3:
        start_time, restart_every, condition_id = data
    else:
        raise ValueError(
            "Too many period delimiters `{PERIOD_DELIMITER}` in period data "
            f"`{data}`."
        )

    try:
        start_time = float(start_time)
        restart_every = float(restart_every)
    except TypeError as e:
        raise ValueError(
            "Currently, only fixed values are allow for the startTime and "
            "restartEvery definitions of a period. Non-fixed values were "
            f"seen in the period definition `{period_def}`."
        ) from e

    return start_time, restart_every, condition_id


def toposort_experiments(experiment_df: pd.DataFrame) -> list[str]:
    """Order experiments according to nested dependencies.

    The returned list will be similar to the following process. For each
    pairing of experiment IDs, "experiment_A" and "experiment_B", if
    "experiment_A" is nested in "experiment_B", then "experiment_A"
    will appear earlier in the returned list than "experiment_B".

    Arguments:
        experiment_df:
            The experiment table.

    Returns:
        The experiment IDs, ordered from no dependencies to most dependencies.
    """
    experiment_ids = sorted(experiment_df.index)
    dependencies = {
        experiment_id: {
            # The experiment contains a nested experiment if one of its periods
            # has a condition that is actually a experiment ID.
            condition_id
            for period in (
                experiment_df.loc[experiment_id, EXPERIMENT].split(
                    EXPERIMENT_DELIMITER
                )
            )
            if (condition_id := period.split(PERIOD_DELIMITER)[-1])
            in experiment_ids
        }
        for experiment_id in experiment_ids
    }
    dependencies = list(TopologicalSorter(graph=dependencies).static_order())
    return dependencies


# FIXME lint:
# experiment IDs do not match any condition IDs
# measurement_df experiment IDs are empty or exist in experiment table
# no circular dependencies from nested experiments
# experiment ID is valid petab ID
# experiment definition matches format
#     `time[:restartEvery]:conditionId|experimentId;...`
# a nested experiment does not involve preequilibration
