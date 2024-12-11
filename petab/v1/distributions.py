"""Probability distributions used by PEtab."""
from __future__ import annotations

import abc

import numpy as np
from scipy.stats import laplace, norm, uniform

__all__ = [
    "Distribution",
    "Normal",
    "Uniform",
    "Laplace",
]


class Distribution(abc.ABC):
    """A univariate probability distribution.

    This class provides a common interface for sampling from and evaluating
    the probability density function of a univariate probability distribution.

    The distribution can be transformed by applying a logarithm to the samples
    and the PDF. This is useful, e.g., for log-normal distributions.

    :param log: If ``True``, the distribution is transformed to its
        corresponding log distribution (e.g., Normal -> LogNormal).
        If a float, the distribution is transformed to its corresponding
        log distribution with the given base (e.g., Normal -> Log10Normal).
        If ``False``, no transformation is applied.
    """

    def __init__(self, log: bool | float = False):
        if log is True:
            log = np.exp(1)
        self._logbase = log

    def _undo_log(self, x: np.ndarray | float) -> np.ndarray | float:
        """Undo the log transformation.

        :param x: The sample to transform.
        :return: The transformed sample
        """
        if self._logbase is False:
            return x
        return self._logbase**x

    def _apply_log(self, x: np.ndarray | float) -> np.ndarray | float:
        """Apply the log transformation.

        :param x: The value to transform.
        :return: The transformed value.
        """
        if self._logbase is False:
            return x
        return np.log(x) / np.log(self._logbase)

    def sample(self, shape=None) -> np.ndarray:
        """Sample from the distribution.

        :param shape: The shape of the sample.
        :return: A sample from the distribution.
        """
        sample = self._sample(shape)
        return self._undo_log(sample)

    @abc.abstractmethod
    def _sample(self, shape=None) -> np.ndarray:
        """Sample from the underlying distribution.

        :param shape: The shape of the sample.
        :return: A sample from the underlying distribution,
            before applying, e.g., the log transformation.
        """
        ...

    def pdf(self, x):
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
        :return: The value of the PDF at ``x``.
        """
        # handle the log transformation; see also:
        #  https://en.wikipedia.org/wiki/Probability_density_function#Scalar_to_scalar
        chain_rule_factor = (
            (1 / (x * np.log(self._logbase))) if self._logbase else 1
        )
        return self._pdf(self._apply_log(x)) * chain_rule_factor

    @abc.abstractmethod
    def _pdf(self, x):
        """Probability density function of the underlying distribution at x.

        :param x: The value at which to evaluate the PDF.
        :return: The value of the PDF at ``x``.
        """
        ...

    @property
    def logbase(self) -> bool | float:
        """The base of the log transformation.

        If ``False``, no transformation is applied.
        """
        return self._logbase


class Normal(Distribution):
    """A (log-)normal distribution.

    :param loc: The location parameter of the distribution.
    :param scale: The scale parameter of the distribution.
    :param truncation: The truncation limits of the distribution.
    :param log: If ``True``, the distribution is transformed to a log-normal
        distribution. If a float, the distribution is transformed to a
        log-normal distribution with the given base.
        If ``False``, no transformation is applied.
        If a transformation is applied, the location and scale parameters
        and the truncation limits are the location, scale and truncation limits
        of the underlying normal distribution.
    """

    def __init__(
        self,
        loc: float,
        scale: float,
        truncation: tuple[float, float] | None = None,
        log: bool | float = False,
    ):
        super().__init__(log=log)
        self._loc = loc
        self._scale = scale
        self._truncation = truncation

        if truncation is not None:
            raise NotImplementedError("Truncation is not yet implemented.")

    def __repr__(self):
        trunc = f", truncation={self._truncation}" if self._truncation else ""
        log = f", log={self._logbase}" if self._logbase else ""
        return f"Normal(loc={self._loc}, scale={self._scale}{trunc}{log})"

    def _sample(self, shape=None):
        return np.random.normal(loc=self._loc, scale=self._scale, size=shape)

    def _pdf(self, x):
        return norm.pdf(x, loc=self._loc, scale=self._scale)

    @property
    def loc(self):
        """The location parameter of the underlying distribution."""
        return self._loc

    @property
    def scale(self):
        """The scale parameter of the underlying distribution."""
        return self._scale


class Uniform(Distribution):
    """A (log-)uniform distribution.

    :param low: The lower bound of the distribution.
    :param high: The upper bound of the distribution.
    :param log: If ``True``, the distribution is transformed to a log-uniform
        distribution. If a float, the distribution is transformed to a
        log-uniform distribution with the given base.
        If ``False``, no transformation is applied.
        If a transformation is applied, the lower and upper bounds are the
        lower and upper bounds of the underlying uniform distribution.
    """

    def __init__(
        self,
        low: float,
        high: float,
        *,
        log: bool | float = False,
    ):
        super().__init__(log=log)
        self._low = low
        self._high = high

    def __repr__(self):
        log = f", log={self._logbase}" if self._logbase else ""
        return f"Uniform(low={self._low}, high={self._high}{log})"

    def _sample(self, shape=None):
        return np.random.uniform(low=self._low, high=self._high, size=shape)

    def _pdf(self, x):
        return uniform.pdf(x, loc=self._low, scale=self._high - self._low)


class Laplace(Distribution):
    """A (log-)Laplace distribution.

    :param loc: The location parameter of the distribution.
    :param scale: The scale parameter of the distribution.
    :param truncation: The truncation limits of the distribution.
    :param log: If ``True``, the distribution is transformed to a log-Laplace
        distribution. If a float, the distribution is transformed to a
        log-Laplace distribution with the given base.
        If ``False``, no transformation is applied.
        If a transformation is applied, the location and scale parameters
        and the truncation limits are the location, scale and truncation limits
        of the underlying Laplace distribution.
    """

    def __init__(
        self,
        loc: float,
        scale: float,
        truncation: tuple[float, float] | None = None,
        log: bool | float = False,
    ):
        super().__init__(log=log)
        self._loc = loc
        self._scale = scale
        self._truncation = truncation
        if truncation is not None:
            raise NotImplementedError("Truncation is not yet implemented.")

    def __repr__(self):
        trunc = f", truncation={self._truncation}" if self._truncation else ""
        log = f", log={self._logbase}" if self._logbase else ""
        return f"Laplace(loc={self._loc}, scale={self._scale}{trunc}{log})"

    def _sample(self, shape=None):
        return np.random.laplace(loc=self._loc, scale=self._scale, size=shape)

    def _pdf(self, x):
        return laplace.pdf(x, loc=self._loc, scale=self._scale)

    @property
    def loc(self):
        """The location parameter of the underlying distribution."""
        return self._loc

    @property
    def scale(self):
        """The scale parameter of the underlying distribution."""
        return self._scale
