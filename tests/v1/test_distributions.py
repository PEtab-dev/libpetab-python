import sys
from math import exp

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


def test_log_uniform():
    """Test Uniform(a, b, log=True) vs LogUniform(a, b)."""
    # support between exp(1) and exp(2)
    dist = Uniform(1, 2, log=True)
    assert dist.pdf(exp(0)) == 0
    assert dist.pdf(exp(1)) > 0
    assert dist.pdf(exp(2)) > 0
    assert dist.pdf(exp(3)) == 0

    # support between 1 and 2
    dist = LogUniform(1, 2)
    assert dist.pdf(0) == 0
    assert dist.pdf(1) > 0
    assert dist.pdf(2) > 0
    assert dist.pdf(3) == 0


@pytest.mark.parametrize("trunc", [(-np.inf, np.inf), [-np.inf, np.inf]])
def test_infinite_truncation_is_no_truncation(trunc):
    """Truncation at (-inf, inf) is equivalent to no truncation."""
    d = Normal(0, 1, trunc=trunc)
    assert d._trunc is None
    assert repr(d) == repr(Normal(0, 1))


def test_trunc_sequence():
    """Truncation limits may be given as any sequence."""
    d = Normal(0, 1, trunc=[-1, 1])
    assert d._trunc == (-1, 1)
    assert d.trunc_low == -1
    assert d.trunc_high == 1


@pytest.mark.parametrize(
    "low, high",
    [
        (0, np.inf),
        (-np.inf, 0),
        (-np.inf, np.inf),
        (np.nan, 1),
        (1, 1),
        (2, 1),
    ],
)
@pytest.mark.parametrize("log", [False, True, 10])
def test_uniform_invalid_limits(low, high, log):
    """Uniform distributions require finite limits with low < high."""
    with pytest.raises(ValueError, match="finite limits"):
        Uniform(low, high, log=log)


@pytest.mark.parametrize(
    "low, high", [(0, 1), (-1, 1), (1, np.inf), (np.nan, 1), (1, 1), (2, 1)]
)
def test_log_uniform_invalid_limits(low, high):
    """Log-uniform distributions require 0 < low < high < inf."""
    with pytest.raises(ValueError, match="0 < low < high < inf"):
        LogUniform(low, high)


def test_truncated_cdf_ppf():
    """Test PDF, CDF and PPF of truncated distributions outside the support
    and for NaN input."""
    from scipy.stats import truncnorm

    dist = Normal(0, 1, trunc=(-1, 0))
    x = [-2, -1, -0.5, 0, 1, np.nan]
    assert_allclose(dist.cdf(x), truncnorm.cdf(x, -1, 0), atol=1e-14)
    assert_allclose(dist.pdf(x), truncnorm.pdf(x, -1, 0), atol=1e-14)
    assert dist.cdf(-2) == 0
    assert dist.cdf(-1) == 0
    assert dist.cdf(0) == 1
    assert dist.cdf(1) == 1
    assert 0 < dist.cdf(-0.5) < 1
    assert np.isnan(dist.cdf(np.nan))
    assert np.isnan(dist.pdf(np.nan))

    q = [-0.1, 0, 0.5, 1, 1.5]
    assert_allclose(dist.ppf(q), truncnorm.ppf(q, -1, 0), atol=1e-14)
    assert np.isnan(dist.ppf(-0.1))
    assert np.isnan(dist.ppf(1.5))
    assert -1 < dist.ppf(0.5) < 0

    dist = Normal(0, 1, log=True, trunc=(1, 2))
    assert_allclose(dist.cdf([-1, 0, 0.5, 1, 2, 3]), [0, 0, 0, 0, 1, 1])
    assert 0 < dist.cdf(1.5) < 1
    assert np.isnan(dist.pdf(np.nan))
    assert np.isnan(dist.ppf(1.5))
    assert 1 < dist.ppf(0.5) < 2


@pytest.mark.parametrize(
    "distribution",
    [
        Normal(0, 1),
        Normal(0, 1, log=True),
        Normal(0, 1, trunc=(-1, 2)),
        Normal(0, 1, log=True, trunc=(1, 2)),
        Uniform(0, 2, log=10),
        Exponential(1, trunc=(0, 2)),
    ],
)
@pytest.mark.parametrize("method", ["pdf", "cdf", "ppf"])
def test_scalar_and_array_output(distribution, method):
    """Scalar input gives scalar output; sequences give arrays."""
    f = getattr(distribution, method)
    x = 0.7 if method == "ppf" else 1.5

    res = f(x)
    assert isinstance(res, float) and not isinstance(res, np.ndarray)
    assert np.isfinite(res) and res > 0

    for xs in ([x, x], np.array([x, x])):
        res_arr = f(xs)
        assert isinstance(res_arr, np.ndarray)
        assert res_arr.shape == (2,)
        assert_allclose(res_arr, res)
    assert f(np.array([[x]])).shape == (1, 1)
