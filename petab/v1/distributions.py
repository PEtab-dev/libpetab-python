"""Probability distributions used by PEtab."""
from __future__ import annotations

import abc

import numpy as np
from scipy.stats import laplace, lognorm, loguniform, norm, uniform

__all__ = [
    "Distribution",
    "Normal",
    "LogNormal",
    "Uniform",
    "LogUniform",
    "Laplace",
    "LogLaplace",
]


class Distribution(abc.ABC):
    """A univariate probability distribution."""

    @abc.abstractmethod
    def sample(self, shape=None) -> np.ndarray:
        """Sample from the distribution.

        :param shape: The shape of the sample.
        :return: A sample from the distribution.
        """
        ...

    @abc.abstractmethod
    def pdf(self, x):
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
        :return: The value of the PDF at ``x``.
        """
        ...


class Normal(Distribution):
    """A normal distribution."""

    def __init__(
        self,
        mean: float,
        std: float,
        truncation: tuple[float, float] | None = None,
    ):
        super().__init__()
        self._mean = mean
        self._std = std
        self._truncation = truncation

        if truncation is not None:
            raise NotImplementedError("Truncation is not yet implemented.")

    def __repr__(self):
        return (
            f"Normal(mean={self._mean}, std={self._std}, "
            f"truncation={self._truncation})"
        )

    def sample(self, shape=None):
        return np.random.normal(loc=self._mean, scale=self._std, size=shape)

    def pdf(self, x):
        return norm.pdf(x, loc=self._mean, scale=self._std)


class LogNormal(Distribution):
    """A log-normal distribution.

    :param mean: The mean of the underlying normal distribution.
    :param std: The standard deviation of the underlying normal distribution.

    """

    def __init__(
        self,
        mean: float,
        std: float,
        truncation: tuple[float, float] | None = None,
        base: float = np.exp(1),
    ):
        super().__init__()
        self._mean = mean
        self._std = std
        self._truncation = truncation
        self._base = base

        if truncation is not None:
            raise NotImplementedError("Truncation is not yet implemented.")

        if base != np.exp(1):
            raise NotImplementedError("Only base e is supported.")

    def __repr__(self):
        return (
            f"LogNormal(mean={self._mean}, std={self._std}, "
            f"base={self._base}, truncation={self._truncation})"
        )

    def sample(self, shape=None):
        return np.random.lognormal(
            mean=self._mean, sigma=self._std, size=shape
        )

    def pdf(self, x):
        return lognorm.pdf(x, scale=np.exp(self._mean), s=self._std)


class Uniform(Distribution):
    """A uniform distribution."""

    def __init__(
        self,
        low: float,
        high: float,
    ):
        super().__init__()
        self._low = low
        self._high = high

    def __repr__(self):
        return f"Uniform(low={self._low}, high={self._high})"

    def sample(self, shape=None):
        return np.random.uniform(low=self._low, high=self._high, size=shape)

    def pdf(self, x):
        return uniform.pdf(x, loc=self._low, scale=self._high - self._low)


class LogUniform(Distribution):
    """A log-uniform distribution.

    :param low: The lower bound of the underlying normal distribution.
    :param high: The upper bound of the underlying normal distribution.
    """

    def __init__(
        self,
        low: float,
        high: float,
        base: float = np.exp(1),
    ):
        super().__init__()
        self._low = low
        self._high = high
        self._base = base
        # re-scaled distribution parameters as required by
        #  scipy.stats.loguniform
        self._low_internal = np.exp(np.log(base) * low)
        self._high_internal = np.exp(np.log(base) * high)

    def __repr__(self):
        return (
            f"LogUniform(low={self._low}, high={self._high}, "
            f"base={self._base})"
        )

    def sample(self, shape=None):
        return loguniform.rvs(
            self._low_internal, self._high_internal, size=shape
        )

    def pdf(self, x):
        return loguniform.pdf(x, self._low_internal, self._high_internal)


class Laplace(Distribution):
    """A Laplace distribution."""

    def __init__(
        self,
        loc: float,
        scale: float,
        truncation: tuple[float, float] | None = None,
    ):
        super().__init__()
        self._loc = loc
        self._scale = scale
        self._truncation = truncation
        if truncation is not None:
            raise NotImplementedError("Truncation is not yet implemented.")

    def sample(self, shape=None):
        return np.random.laplace(loc=self._loc, scale=self._scale, size=shape)

    def pdf(self, x):
        return laplace.pdf(x, loc=self._loc, scale=self._scale)


class LogLaplace(Distribution):
    """A log-Laplace distribution."""

    def __init__(
        self,
        loc: float,
        scale: float,
        truncation: tuple[float, float] | None = None,
        base: float = np.exp(1),
    ):
        super().__init__()
        self._loc = loc
        self._scale = scale
        self._truncation = truncation
        self._base = base
        if truncation is not None:
            raise NotImplementedError("Truncation is not yet implemented.")
        if base != np.exp(1):
            raise NotImplementedError("Only base e is supported.")

    def __repr__(self):
        return (
            f"LogLaplace(loc={self._loc}, scale={self._scale}, "
            f"base={self._base}, truncation={self._truncation})"
        )

    @property
    def loc(self):
        """The mean of the underlying Laplace distribution."""
        return self._loc

    @property
    def scale(self):
        """The scale of the underlying Laplace distribution."""
        return self._scale

    def sample(self, shape=None):
        return np.exp(
            np.random.laplace(loc=self._loc, scale=self._scale, size=shape)
        )

    def pdf(self, x):
        return (
            1
            / (2 * self.scale * x)
            * np.exp(-np.abs(np.log(x) - self._loc) / self._scale)
        )
