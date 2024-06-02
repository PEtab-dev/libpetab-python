"""Functions operating on the PEtab timecouse table."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd
from graphlib import TopologicalSorter
from more_itertools import one

from . import core, lint, measurements
from .C import (
    EXPERIMENT,
    EXPERIMENT_DELIMITER,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PERIOD_DELIMITER,
    TIME,
    TIME_STEADY_STATE,
)

__all__ = [
    "get_experiment_df",
    "write_experiment_df",
    "Timecourse",
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
            f"Timecourse table missing mandatory field {EXPERIMENT_ID}."
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


@dataclass
class Period:
    """A experiment period.

    Attributes:
        condition_id:
            The PEtab condition ID.
        end_time:
            The time when the period ends.
        start_time:
            The time when the period starts.
        experiment_id:
            The ID of the experiment that this period belongs to.
    """

    condition_id: str
    end_time: Time
    start_time: Time
    experiment_id: str

    def get_condition(self, condition_df: pd.DataFrame) -> pd.Series:
        return condition_df.loc[self.condition_id]

    def get_measurements(
        self,
        measurement_df: pd.DataFrame,
        include_end: bool = False,
    ) -> pd.Series:
        experiment_measurement_df = measurements.get_experiment_measurements(
            measurement_df=measurement_df, experiment_id=self.experiment_id
        )

        after_start = experiment_measurement_df[TIME] >= self.start_time
        before_end = (
            experiment_measurement_df[TIME] <= self.end_time
            if include_end
            else experiment_measurement_df[TIME] < self.end_time
        )
        return experiment_measurement_df.loc[after_start & before_end]


@dataclass
class Timecourse:
    """A experiment.


    Attributes:
        name:
            The experiment name.
        periods:
            The periods of the experiment.
        #t0:
        #    The time when the experiment starts.
        experiment_id:
            The experiment ID.
    """

    experiment_id: str
    periods: list[Period]
    name: None | str = None

    @property
    def timepoints(self):
        _timepoints = [self.t0]
        # skip last since it has indefinite duration
        for period in self.periods[:-1]:
            _timepoints.append(_timepoints[-1] + period.duration)
        return _timepoints

    @property
    def condition_ids(self):
        return [period.condition_id for period in self.periods]

    def to_series(self):
        period_definitions = []
        for period in self.periods:
            period_definition = PERIOD_DELIMITER.join(
                [
                    str(period.start_time),
                    period.condition_id,
                ]
            )
            period_definitions.append(period_definition)
        experiment_definition = EXPERIMENT_DELIMITER.join(period_definitions)

        series = pd.Series(
            data={EXPERIMENT: experiment_definition},
            name=self.experiment_id,
        )
        return series

    @staticmethod
    def from_df_row(
        row: pd.Series,
        # measurement_df: pd.DataFrame = None,
        experiments: dict[str, Timecourse] = None,
    ) -> Timecourse:
        """Create a experiment object from a row definition.

        Any nested or repetitive structure is flattened.

        Argments:
            row:
                The row definition.
            experiments:
                Required to flatten nested experiments. Keys are experiment
                IDs.

        Returns:
            The experiment.
        """
        experiment_id = row.name

        # Collect all periods, which are possibly nested in one of two ways:
        #    1. `restartEvery` is specified
        #    2. another experimentId is used as the condition for a period
        # We use "true period" to refer to a period as it is seen in a
        # experiment table, i.e. as truth as specified by the user. We use
        # "denested period" to refer to one of the possibly-many periods that
        # actually occur within the true period, after it is denested.
        true_periods = []
        for period_definition in row.get(EXPERIMENT).split(
            EXPERIMENT_DELIMITER
        ):
            true_periods.append(_parse_period_definition(period_definition))

        # Denest any actually nested periods
        denested_periods = []
        for true_period_index, true_period in enumerate(true_periods):
            (
                true_start_time,
                restart_every,
                true_period_condition_id,
            ) = true_period

            # Determine the end time of the current period.
            # The period ends when the next period starts, or never.
            try:
                true_end_time = true_periods[true_period_index + 1][0]
            except IndexError:
                true_end_time = TIME_STEADY_STATE

            # TODO for now, require fixed time points for proper denesting
            try:
                true_start_time = float(true_start_time)
                true_end_time = float(true_end_time)
            except ValueError as e:
                raise ValueError(
                    "Parameterized experiment times are not yet supported. "
                    f"In experiment {experiment_id}, there is a period "
                    f"`{true_period_index}` starts at `{true_start_time}` and "
                    f"ends at `{true_end_time}`."
                ) from e

            # TODO for now, require finite period durations if `restartEvery`
            # nesting is specified
            if restart_every and (
                core.time_is_at_steady_state(
                    true_start_time, preequilibration=True
                )
                or core.time_is_at_steady_state(
                    true_end_time, postequilibration=True
                )
            ):
                raise ValueError(
                    "Period restarts are currently not supported in "
                    "periods that are simulated until steady-state. This "
                    f"occurs in experiment `{experiment_id}` during period "
                    f"`{true_period_index}`. Period start time: "
                    f"`{true_start_time}`. Period end time: `{true_end_time}`."
                )

            # Create lists of start and end times for each period repeat
            # due to `restartEvery` nesting
            restart_end_times = []
            # Add initial period start
            restart_start_times = [true_start_time]
            if restart_every:
                next_restart = true_start_time
                while (
                    next_restart := next_restart + restart_every
                ) < true_end_time:
                    # Add end of previous period restart
                    restart_end_times.append(next_restart)
                    # Add start of next period restart
                    restart_start_times.append(next_restart)
            # Add final period end
            restart_end_times.append(true_end_time)

            # Generate a new period for each restart due to `restartEvery`
            # nesting
            for restart_start_time, restart_end_time in zip(
                restart_start_times, restart_end_times, strict=True
            ):
                denested_period = [
                    Period(
                        start_time=restart_start_time,
                        end_time=restart_end_time,
                        condition_id=true_period_condition_id,
                        experiment_id=experiment_id,
                    )
                ]

                # Handle `experimentId` condition nesting
                if nested_experiment := experiments.get(
                    true_period_condition_id
                ):
                    denested_period = copy.deepcopy(nested_experiment.periods)

                    # TODO for now, require this
                    if denested_period[0].start_time != 0:
                        raise ValueError(
                            "The first period of a nested experiment must "
                            "have a start time of `0`. The experiment "
                            f"`{nested_experiment.experiment_id}` is nested "
                            f"within experiment `{experiment_id}`."
                        )

                    # Shift all nested periods in time, to match the start time
                    # of the period restart. Drop nested periods that start
                    # after the current period restart ends.
                    current_start_time = restart_start_time
                    for nested_experiment_period in denested_period:
                        nested_experiment_period.experiment_id = experiment_id

                        nested_experiment_period.start_time = (
                            current_start_time
                        )
                        nested_experiment_period.end_time += current_start_time

                        if (
                            nested_experiment_period.end_time
                            > restart_end_time
                        ):
                            nested_experiment_period.end_time = (
                                restart_end_time
                            )
                            # Drop nested periods if they will start after the
                            # current restarted period is scheduled to end.
                            break

                        current_start_time = nested_experiment_period.end_time

                denested_periods.extend(denested_period)

        return Timecourse(
            experiment_id=experiment_id,
            name=row.get(EXPERIMENT_NAME),
            periods=denested_periods,
        )

    @staticmethod
    def from_df(
        experiment_df: pd.DataFrame,
        experiment_ids: str | list[str] = None,
    ) -> dict[str, Timecourse] | Timecourse:
        """Read in all experiment(s).

        Arguments:
            experiment_df:
                The PEtab experiment table.
            experiment_ids:
                Specific experiment id(s).

        Returns:
            The experiment(s).
        """
        experiment_df = _clean_experiment_df(experiment_df)

        # TODO performance gain: only return experiments required to fully
        # denest the supplied `experiment_ids` experiments. Currently returns
        # all experiments in an order that supports denesting.
        sorted_experiment_ids = toposort_experiments(experiment_df)

        experiments = {}
        for experiment_id, row in experiment_df.loc[
            sorted_experiment_ids
        ].iterrows():
            experiments[experiment_id] = Timecourse.from_df_row(
                row=row,
                experiments=experiments,
            )

        if experiment_ids is None:
            experiment_ids = sorted_experiment_ids
        if isinstance(experiment_ids, str):
            experiment_ids = [experiment_ids]
        experiments = {
            experiment_id: experiment
            for experiment_id, experiment in experiments.items()
            if experiment_id in experiment_ids
        }

        if len(experiment_ids) == 1:
            experiments = experiments[one(experiment_ids)]

        return experiments

    def __len__(self):
        return len(self.periods)


def _clean_experiment_df(experiment_df: pd.DataFrame) -> str:
    experiment_df[EXPERIMENT] = experiment_df[EXPERIMENT].apply(
        lambda experiment_definition: experiment_definition.replace(" ", "")
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
