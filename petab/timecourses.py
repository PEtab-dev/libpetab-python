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
    PERIOD_DELIMITER,
    TIME,
    TIME_STEADY_STATE,
    TIMECOURSE,
    TIMECOURSE_DELIMITER,
    TIMECOURSE_ID,
    TIMECOURSE_NAME,
)

__all__ = [
    "get_timecourse_df",
    "write_timecourse_df",
    "Timecourse",
    "Period",
]


Time = TypeVar("Time", float, int, str)
UpdateValue = TypeVar("UpdateValue", float, int, str)


def get_timecourse_df(timecourse_file: str | Path | None) -> pd.DataFrame:
    """Read the provided timecourse file into a ``pandas.Dataframe``.

    Arguments:
        timecourse_file:
            Location of PEtab timecourse file, or a ``pandas.Dataframe``.

    Returns:
        The timecourses dataframe.
    """
    if timecourse_file is None:
        return timecourse_file

    if isinstance(timecourse_file, (str, Path)):
        timecourse_file = pd.read_csv(
            timecourse_file,
            sep="\t",
            float_precision="round_trip",
        )

    lint.assert_no_leading_trailing_whitespace(
        timecourse_file.columns.values, TIMECOURSE
    )

    if not isinstance(timecourse_file.index, pd.RangeIndex):
        timecourse_file.reset_index(inplace=True)

    try:
        timecourse_file.set_index([TIMECOURSE_ID], inplace=True)
    except KeyError as e:
        raise KeyError(
            f"Timecourse table missing mandatory field {TIMECOURSE_ID}."
        ) from e

    return timecourse_file


def write_timecourse_df(df: pd.DataFrame, filename: str | Path) -> None:
    """Write the provided PEtab timecourse table to disk.

    Arguments:
        df:
            The PEtab timecourse table.
        filename:
            The table will be written to this location.
    """
    df.to_csv(filename, sep="\t", index=True)


@dataclass
class Period:
    """A timecourse period.

    Attributes:
        condition_id:
            The PEtab condition ID.
        end_time:
            The time when the period ends.
        start_time:
            The time when the period starts.
        timecourse_id:
            The ID of the timecourse that this period belongs to.
    """

    condition_id: str
    end_time: Time
    start_time: Time
    timecourse_id: str

    def get_condition(self, condition_df: pd.DataFrame) -> pd.Series:
        return condition_df.loc[self.condition_id]

    def get_measurements(
        self,
        measurement_df: pd.DataFrame,
        include_end: bool = False,
    ) -> pd.Series:
        timecourse_measurement_df = measurements.get_timecourse_measurements(
            measurement_df=measurement_df, timecourse_id=self.timecourse_id
        )

        after_start = timecourse_measurement_df[TIME] >= self.start_time
        before_end = (
            timecourse_measurement_df[TIME] <= self.end_time
            if include_end
            else timecourse_measurement_df[TIME] < self.end_time
        )
        return timecourse_measurement_df.loc[after_start & before_end]


@dataclass
class Timecourse:
    """A timecourse.


    Attributes:
        name:
            The timecourse name.
        periods:
            The periods of the timecourse.
        #t0:
        #    The time when the timecourse starts.
        timecourse_id:
            The timecourse ID.
    """

    timecourse_id: str
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

    # def to_df(self):
    #     data = {
    #         TIMECOURSE_ID: [self.timecourse_id],
    #         TIMECOURSE_NAME: [self.name],
    #         TIMECOURSE: [None],
    #     }
    #     if self.name is None:
    #         del data[TIMECOURSE_NAME]

    #     t0 = self.t0
    #     times = []
    #     condition_ids = []
    #     for period in self.periods:
    #         times.append(t0)
    #         condition_ids.append(period.condition_id)
    #         t0 += period.duration

    #     timecourse_str = PERIOD_DELIMITER.join(
    #         TIME_CONDITION_DELIMITER.join([str(time), condition_id])
    #         for time, condition_id in zip(times, condition_ids)
    #     )
    #     data[TIMECOURSE] = timecourse_str

    #     return get_timecourse_df(
    #         pd.DataFrame(data=data)
    #     )

    @staticmethod
    def from_df_row(
        row: pd.Series,
        # measurement_df: pd.DataFrame = None,
        timecourses: dict[str, Timecourse] = None,
    ) -> Timecourse:
        """Create a timecourse object from a row definition.

        Any nested or repetitive structure is flattened.

        Argments:
            row:
                The row definition.
            timecourses:
                Required to flatten nested timecourses. Keys are timecourse
                IDs.

        Returns:
            The timecourse.
        """
        timecourse_id = row.name

        # Collect all periods, which are possibly nested in one of two ways:
        #    1. `restartEvery` is specified
        #    2. another timecourseId is used as the condition for a period
        # We use "true period" to refer to a period as it is seen in a
        # timecourse table, i.e. as truth as specified by the user. We use
        # "denested period" to refer to one of the possibly-many periods that
        # actually occur within the true period, after it is denested.
        true_periods = []
        for period_definition in row.get(TIMECOURSE).split(
            TIMECOURSE_DELIMITER
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
                    "Parameterized timecourse times are not yet supported. "
                    f"In timecourse {timecourse_id}, there is a period "
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
                    f"occurs in timecourse `{timecourse_id}` during period "
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
                        timecourse_id=timecourse_id,
                    )
                ]

                # Handle `timecourseId` condition nesting
                if nested_timecourse := timecourses.get(
                    true_period_condition_id
                ):
                    denested_period = copy.deepcopy(nested_timecourse.periods)

                    # TODO for now, require this
                    if denested_period[0].start_time != 0:
                        raise ValueError(
                            "The first period of a nested timecourse must "
                            "have a start time of `0`. The timecourse "
                            f"`{nested_timecourse.timecourse_id}` is nested "
                            f"within timecourse `{timecourse_id}`."
                        )

                    # Shift all nested periods in time, to match the start time
                    # of the period restart. Drop nested periods that start
                    # after the current period restart ends.
                    current_start_time = restart_start_time
                    for nested_timecourse_period in denested_period:
                        nested_timecourse_period.timecourse_id = timecourse_id

                        nested_timecourse_period.start_time = (
                            current_start_time
                        )
                        nested_timecourse_period.end_time += current_start_time

                        if (
                            nested_timecourse_period.end_time
                            > restart_end_time
                        ):
                            nested_timecourse_period.end_time = (
                                restart_end_time
                            )
                            # Drop nested periods if they will start after the
                            # current restarted period is scheduled to end.
                            break

                        current_start_time = nested_timecourse_period.end_time

                denested_periods.extend(denested_period)

        return Timecourse(
            timecourse_id=timecourse_id,
            name=row.get(TIMECOURSE_NAME),
            periods=denested_periods,
        )

    @staticmethod
    def from_df(
        timecourse_df: pd.DataFrame,
        timecourse_ids: str | list[str] = None,
    ) -> dict[str, Timecourse] | Timecourse:
        """Read in all timecourse(s).

        Arguments:
            timecourse_df:
                The PEtab timecourse table.
            timecourse_ids:
                Specific timecourse id(s).

        Returns:
            The timecourse(s).
        """
        timecourse_df = _clean_timecourse_df(timecourse_df)

        # TODO performance gain: only return timecourses required to fully
        # denest the supplied `timecourse_ids` timecourses. Currently returns
        # all timecourses in an order that supports denesting.
        sorted_timecourse_ids = toposort_timecourses(timecourse_df)

        timecourses = {}
        for timecourse_id, row in timecourse_df.loc[
            sorted_timecourse_ids
        ].iterrows():
            timecourses[timecourse_id] = Timecourse.from_df_row(
                row=row,
                timecourses=timecourses,
            )

        if timecourse_ids is None:
            timecourse_ids = sorted_timecourse_ids
        if isinstance(timecourse_ids, str):
            timecourse_ids = [timecourse_ids]
        timecourses = {
            timecourse_id: timecourse
            for timecourse_id, timecourse in timecourses.items()
            if timecourse_id in timecourse_ids
        }

        if len(timecourse_ids) == 1:
            timecourses = timecourses[one(timecourse_ids)]

        return timecourses

    def __len__(self):
        return len(self.periods)


def _clean_timecourse_df(timecourse_df: pd.DataFrame) -> str:
    timecourse_df[TIMECOURSE] = timecourse_df[TIMECOURSE].apply(
        lambda timecourse_definition: timecourse_definition.replace(" ", "")
    )
    return timecourse_df


def _parse_period_definition(period_def: str) -> list[float | str]:
    """Parse a period definition.

    The expected definition format is:

        startTime[:restartEvery]:conditionId

    Arguments:
        period_def:
            The period definition, in the same format as specified in the
            timecourse table.

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


def toposort_timecourses(timecourse_df: pd.DataFrame) -> list[str]:
    """Order timecourses according to nested dependencies.

    The returned list will be similar to the following process. For each
    pairing of timecourse IDs, "timecourse_A" and "timecourse_B", if
    "timecourse_A" is nested in "timecourse_B", then "timecourse_A"
    will appear earlier in the returned list than "timecourse_B".

    Arguments:
        timecourse_df:
            The timecourse table.

    Returns:
        The timecourse IDs, ordered from no dependencies to most dependencies.
    """
    timecourse_ids = sorted(timecourse_df.index)
    dependencies = {
        timecourse_id: {
            # The timecourse contains a nested timecourse if one of its periods
            # has a condition that is actually a timecourse ID.
            condition_id
            for period in (
                timecourse_df.loc[timecourse_id, TIMECOURSE].split(
                    TIMECOURSE_DELIMITER
                )
            )
            if (condition_id := period.split(PERIOD_DELIMITER)[-1])
            in timecourse_ids
        }
        for timecourse_id in timecourse_ids
    }
    dependencies = list(TopologicalSorter(graph=dependencies).static_order())
    return dependencies


# FIXME lint:
# timecourse IDs do not match any condition IDs
# measurement_df timecourse IDs are empty or exist in timecourse table
# no circular dependencies from nested timecourses
# timecourse ID is valid petab ID
# timecourse definition matches format
#     `time[:restartEvery]:conditionId|timecourseId;...`
# a nested timecourse does not involve preequilibration
