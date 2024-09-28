"""PEtab v2 core functions"""

import math

from ..v1.core import *
from .C import *


__all__ = [
    "time_is_at_steady_state",
]

POSSIBLE_GROUPVARS_FLATTENED_PROBLEM = [
    OBSERVABLE_ID,
    OBSERVABLE_PARAMETERS,
    NOISE_PARAMETERS,
    EXPERIMENT_ID,
]


# TODO add to COMBINE archive


def time_is_at_steady_state(
    time: float,
    preequilibration: bool = False,
    postequilibration: bool = False,
) -> bool:
    """Check whether a `time` is at steady state or indicates equilibration.

    Both ``preequilibration`` and ``postequilibration`` cannot be ``True``. Set
    both to ``False`` (default) to only check if the time point indicates
    equilibration or a steady-state in general.

    N.B. both preequilibration and postequilibration indicate that the current
    period is simulated until the system reaches a steady-state, not that
    the system is at steady-state at the start time of the period.

    Arguments:
        time:
            The time.
        preequilibration:
            If ``True``, then this method will only return ``True`` if the
            ``time`` indicates preequilibration.
        postequilibration:
            If ``True``, then this method will only return ``True`` if the
            ``time`` indicates postequilibration.

    Returns:
        Whether the time point is at steady state, or optionally indicates
        a specific equilibration type.
    """
    if preequilibration and postequilibration:
        raise ValueError(
            "Only one of `preequilibration` or `postequilibration` can be "
            "set to `True`. See docstring."
        )

    steady_state = math.isinf(time)

    equilibration = True
    if preequilibration and time > 0:
        equilibration = False
    if postequilibration and time < 0:
        equilibration = False

    return steady_state and equilibration
