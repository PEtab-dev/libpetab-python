from itertools import product

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

from petab.v1.distributions import *
from petab.v2.C import *


@pytest.mark.parametrize(
    "distribution, transform",
    list(
        product(
            [
                Normal(1, 1),
                LogNormal(2, 1),
                Uniform(2, 4),
                LogUniform(1, 2),
                Laplace(1, 2),
                LogLaplace(1, 0.5),
            ],
            [LIN, LOG, LOG10],
        )
    ),
)
def test_sample_matches_pdf(distribution, transform):
    """Test that the sample matches the PDF."""
    np.random.seed(1)
    N_SAMPLES = 10_000
    distribution.transform = transform
    sample = distribution.sample(N_SAMPLES)

    # pdf -> cdf
    def cdf(x):
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
