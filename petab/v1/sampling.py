"""Functions related to parameter sampling"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .C import *  # noqa: F403

__all__ = ["sample_from_prior", "sample_parameter_startpoints"]


def sample_from_prior(
    prior: tuple[str, list, str, list], n_starts: int
) -> np.array:
    """Creates samples for one parameter based on prior

    Arguments:
        prior: A tuple as obtained from
            :func:`petab.parameter.get_priors_from_df`
        n_starts: Number of samples

    Returns:
        Array with sampled values
    """
    from .priors import Prior

    # unpack info
    p_type, p_params, scaling, bounds = prior
    prior = Prior(
        p_type,
        tuple(p_params),
        bounds=tuple(bounds),
        transformation=scaling,
    )
    return prior.sample(shape=(n_starts,), x_scaled=True)


def sample_parameter_startpoints(
    parameter_df: pd.DataFrame,
    n_starts: int = 100,
    seed: int = None,
    parameter_ids: Sequence[str] = None,
) -> np.array:
    """Create :class:`numpy.array` with starting points for an optimization

    Arguments:
        parameter_df: PEtab parameter DataFrame
        n_starts: Number of points to be sampled
        seed: Random number generator seed (see :func:`numpy.random.seed`)
        parameter_ids: A sequence of parameter IDs for which to sample starting
            points.
            For subsetting or reordering the parameters.
            Defaults to all estimated parameters.

    Returns:
        Array of sampled starting points with dimensions
        `n_startpoints` x `n_optimization_parameters`
    """
    from .priors import Prior

    if seed is not None:
        np.random.seed(seed)

    par_to_estimate = parameter_df.loc[parameter_df[ESTIMATE] == 1]

    if parameter_ids is not None:
        try:
            par_to_estimate = par_to_estimate.loc[parameter_ids, :]
        except KeyError as e:
            missing_ids = set(parameter_ids) - set(par_to_estimate.index)
            raise KeyError(
                "Parameter table does not contain estimated parameter(s) "
                f"{missing_ids}."
            ) from e

    # get types and parameters of priors from dataframe
    return np.array(
        [
            Prior.from_par_dict(
                row,
                type_="initialization",
            ).sample(n_starts, x_scaled=True)
            for row in par_to_estimate.to_dict("records")
        ]
    ).T
