import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import cumulative_trapezoid
from scipy.stats import (
    kstest,
    laplace,
    loglaplace,
    lognorm,
    loguniform,
    norm,
    uniform,
)

from petab.v1.distributions import *
from petab.v2.C import *


@pytest.mark.parametrize(
    "distribution",
    [
        Normal(2, 1),
        Normal(2, 1, log=True),
        Normal(2, 1, log=10),
        Uniform(2, 4),
        Uniform(-2, 4, log=True),
        Uniform(2, 4, log=10),
        Laplace(1, 2),
        Laplace(1, 0.5, log=True),
        Normal(2, 1, trunc=(1, 2)),
        Normal(2, 1, log=True, trunc=(0.5, 8)),
        Normal(2, 1, log=10),
        Laplace(1, 2, trunc=(1, 2)),
        Laplace(1, 0.5, log=True, trunc=(0.5, 8)),
        Cauchy(2, 1),
        ChiSquare(4),
        Exponential(1),
        Gamma(3, 5),
        Rayleigh(3),
    ],
)
def test_sample_matches_pdf(distribution):
    """Test that the sample matches the PDF."""
    np.random.seed(1)
    N_SAMPLES = 10_000
    sample = distribution.sample(N_SAMPLES)

    def cdf(x):
        # pdf -> cdf
        return cumulative_trapezoid(distribution.pdf(x), x)

    # Kolmogorov-Smirnov test to check if the sample is drawn from the CDF
    _, p = kstest(sample, cdf)

    # if p < 0.05:
    #     import matplotlib.pyplot as plt
    #     plt.hist(sample, bins=100, density=True)
    #     x = np.linspace(min(sample), max(sample), 100)
    #     plt.plot(x, distribution.pdf(x))
    #     plt.show()

    assert p > 0.05, (p, distribution)

    # check min/max of CDF at the bounds
    assert np.isclose(
        distribution.cdf(
            distribution.trunc_low
            if not distribution.logbase
            else max(sys.float_info.min, distribution.trunc_low)
        ),
        0,
        atol=1e-16,
        rtol=0,
    )
    assert np.isclose(
        distribution.cdf(distribution.trunc_high), 1, atol=1e-14, rtol=0
    )

    # Test samples match scipy CDFs
    reference_pdf = None
    if distribution._trunc is None and distribution.logbase is False:
        if isinstance(distribution, Normal):
            reference_pdf = norm.pdf(
                sample, distribution.loc, distribution.scale
            )
        elif isinstance(distribution, Uniform):
            reference_pdf = uniform.pdf(
                sample,
                distribution._low,
                distribution._high - distribution._low,
            )
        elif isinstance(distribution, Laplace):
            reference_pdf = laplace.pdf(
                sample, distribution.loc, distribution.scale
            )

    if distribution._trunc is None and distribution.logbase == np.exp(1):
        if isinstance(distribution, Normal):
            reference_pdf = lognorm.pdf(
                sample, scale=np.exp(distribution.loc), s=distribution.scale
            )
        elif isinstance(distribution, Uniform):
            reference_pdf = loguniform.pdf(
                sample, np.exp(distribution._low), np.exp(distribution._high)
            )
        elif isinstance(distribution, Laplace):
            reference_pdf = loglaplace.pdf(
                sample,
                c=1 / distribution.scale,
                scale=np.exp(distribution.loc),
            )
    if reference_pdf is not None:
        assert_allclose(
            distribution.pdf(sample), reference_pdf, rtol=1e-10, atol=1e-14
        )
