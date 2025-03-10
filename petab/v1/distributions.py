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
    :param trunc: The truncation points (lower, upper) of the distribution
        or ``None`` if the distribution is not truncated.
    """

    def __init__(
        self, *, log: bool | float = False, trunc: tuple[float, float] = None
    ):
        if log is True:
            log = np.exp(1)

        if trunc == (-np.inf, np.inf):
            trunc = None

        if trunc is not None and trunc[0] >= trunc[1]:
            raise ValueError(
                "The lower truncation limit must be smaller "
                "than the upper truncation limit."
            )

        self._logbase = log
        self._trunc = trunc

        self._cd_low = None
        self._cd_high = None
        self._truncation_normalizer = 1

        if self._trunc is not None:
            try:
                # the cumulative density of the transformed distribution at the
                #  truncation limits
                self._cd_low = self._cdf_transformed_untruncated(
                    self.trunc_low
                )
                self._cd_high = self._cdf_transformed_untruncated(
                    self.trunc_high
                )
                # normalization factor for the PDF/CDF of the transformed
                #  distribution to account for truncation
                self._truncation_normalizer = 1 / (
                    self._cd_high - self._cd_low
                )
            except NotImplementedError:
                pass

    @property
    def trunc_low(self) -> float:
        """The lower truncation limit of the transformed distribution."""
        return self._exp(self._trunc[0]) if self._trunc else -np.inf

    @property
    def trunc_high(self) -> float:
        """The upper truncation limit of the transformed distribution."""
        return self._exp(self._trunc[1]) if self._trunc else np.inf

    def _exp(self, x: np.ndarray | float) -> np.ndarray | float:
        """Exponentiate / undo the log transformation if applicable.

        Exponentiate if a log transformation is applied to the distribution.
        Otherwise, return the input.

        :param x: The sample to transform.
        :return: The transformed sample
        """
        if self._logbase is False:
            return x
        return self._logbase**x

    def _log(self, x: np.ndarray | float) -> np.ndarray | float:
        """Apply the log transformation if enabled.

        Compute the log of x with the specified base if a log transformation
        is applied to the distribution. Otherwise, return the input.

        :param x: The value to transform.
        :return: The transformed value.
        """
        if self._logbase is False:
            return x
        return np.log(x) / np.log(self._logbase)

    def sample(self, shape=None) -> np.ndarray | float:
        """Sample from the distribution.

        :param shape: The shape of the sample.
        :return: A sample from the distribution.
        """
        sample = (
            self._exp(self._sample(shape))
            if self._trunc is None
            else self._inverse_transform_sample(shape)
        )

        return sample

    @abc.abstractmethod
    def _sample(self, shape=None) -> np.ndarray | float:
        """Sample from the underlying distribution.

        :param shape: The shape of the sample.
        :return: A sample from the underlying distribution,
            before applying, e.g., the log transformation or truncation.
        """
        ...

    def pdf(self, x) -> np.ndarray | float:
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
        :return: The value of the PDF at ``x``.
        """
        if self._trunc is None:
            return self._pdf_transformed_untruncated(x)

        return np.where(
            (x >= self.trunc_low) & (x <= self.trunc_high),
            self._pdf_transformed_untruncated(x) * self._truncation_normalizer,
            0,
        )

    @abc.abstractmethod
    def _pdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        """Probability density function of the underlying distribution at x.

        :param x: The value at which to evaluate the PDF.
        :return: The value of the PDF at ``x``.
        """
        ...

    def _pdf_transformed_untruncated(self, x) -> np.ndarray | float:
        """Probability density function of the transformed, but untruncated
        distribution at x.

        :param x: The value at which to evaluate the PDF.
        :return: The value of the PDF at ``x``.
        """
        if self.logbase is False:
            return self._pdf_untransformed_untruncated(x)

        # handle the log transformation; see also:
        #  https://en.wikipedia.org/wiki/Probability_density_function#Scalar_to_scalar
        with np.errstate(invalid="ignore", divide="ignore"):
            chain_rule_factor = (
                (1 / (x * np.log(self._logbase))) if self._logbase else 1
            )

            return np.where(
                x > 0,
                self._pdf_untransformed_untruncated(self._log(x))
                * chain_rule_factor,
                0,
            )

    @property
    def logbase(self) -> bool | float:
        """The base of the log transformation.

        If ``False``, no transformation is applied.
        """
        return self._logbase

    def cdf(self, x) -> np.ndarray | float:
        """Cumulative distribution function at x.

        :param x: The value at which to evaluate the CDF.
        :return: The value of the CDF at ``x``.
        """
        if self._trunc is None:
            return self._cdf_transformed_untruncated(x)
        return (
            self._cdf_transformed_untruncated(x) - self._cd_low
        ) * self._truncation_normalizer

    def _cdf_transformed_untruncated(self, x) -> np.ndarray | float:
        """Cumulative distribution function of the transformed, but untruncated
        distribution at x.

        :param x: The value at which to evaluate the CDF.
        :return: The value of the CDF at ``x``.
        """
        if not self.logbase:
            return self._cdf_untransformed_untruncated(x)

        with np.errstate(invalid="ignore"):
            return np.where(
                x < 0, 0, self._cdf_untransformed_untruncated(self._log(x))
            )

    def _cdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        """Cumulative distribution function of the underlying
        (untransformed, untruncated) distribution at x.

        :param x: The value at which to evaluate the CDF.
        :return: The value of the CDF at ``x``.
        """
        raise NotImplementedError

    def _ppf_untransformed_untruncated(self, q) -> np.ndarray | float:
        """Percent point function of the underlying
        (untransformed, untruncated) distribution at q.

        :param q: The quantile at which to evaluate the PPF.
        :return: The value of the PPF at ``q``.
        """
        raise NotImplementedError

    def _ppf_transformed_untruncated(self, q) -> np.ndarray | float:
        """Percent point function of the transformed, but untruncated
        distribution at q.

        :param q: The quantile at which to evaluate the PPF.
        :return: The value of the PPF at ``q``.
        """
        return self._exp(self._ppf_untransformed_untruncated(q))

    def ppf(self, q) -> np.ndarray | float:
        """Percent point function at q.

        :param q: The quantile at which to evaluate the PPF.
        :return: The value of the PPF at ``q``.
        """
        if self._trunc is None:
            return self._ppf_transformed_untruncated(q)

        # Adjust quantiles to account for truncation
        adjusted_q = self._cd_low + q * (self._cd_high - self._cd_low)
        return self._ppf_transformed_untruncated(adjusted_q)

    def _inverse_transform_sample(self, shape) -> np.ndarray | float:
        """Generate an inverse transform sample from the transformed and
        truncated distribution.

        :param shape: The shape of the sample.
        :return: The sample.
        """
        uniform_sample = np.random.uniform(
            low=self._cd_low, high=self._cd_high, size=shape
        )
        return self._ppf_transformed_untruncated(uniform_sample)


class Normal(Distribution):
    """A (log-)normal distribution.

    :param loc: The location parameter of the distribution.
    :param scale: The scale parameter of the distribution.
    :param trunc: The truncation limits of the distribution.
        ``None`` if the distribution is not truncated.
    :param log: If ``True``, the distribution is transformed to a log-normal
        distribution. If a float, the distribution is transformed to a
        log-normal distribution with the given log-base.
        If ``False``, no transformation is applied.

        .. note::

           If a transformation is applied, the location and scale parameters
           and the truncation limits are the location, scale and truncation
           limits of the underlying normal distribution.
    """

    def __init__(
        self,
        loc: float,
        scale: float,
        trunc: tuple[float, float] | None = None,
        log: bool | float = False,
    ):
        self._loc = loc
        self._scale = scale
        super().__init__(log=log, trunc=trunc)

    def __repr__(self):
        trunc = f", trunc={self._trunc}" if self._trunc else ""
        log = f", log={self._logbase}" if self._logbase else ""
        return f"Normal(loc={self._loc}, scale={self._scale}{trunc}{log})"

    def _sample(self, shape=None) -> np.ndarray | float:
        return np.random.normal(loc=self._loc, scale=self._scale, size=shape)

    def _pdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        return norm.pdf(x, loc=self._loc, scale=self._scale)

    def _cdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        return norm.cdf(x, loc=self._loc, scale=self._scale)

    def _ppf_untransformed_untruncated(self, q) -> np.ndarray | float:
        return norm.ppf(q, loc=self._loc, scale=self._scale)

    @property
    def loc(self) -> float:
        """The location parameter of the underlying distribution."""
        return self._loc

    @property
    def scale(self) -> float:
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

        .. note::

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
        self._low = low
        self._high = high
        super().__init__(log=log)

    def __repr__(self):
        log = f", log={self._logbase}" if self._logbase else ""
        return f"Uniform(low={self._low}, high={self._high}{log})"

    def _sample(self, shape=None) -> np.ndarray | float:
        return np.random.uniform(low=self._low, high=self._high, size=shape)

    def _pdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        return uniform.pdf(x, loc=self._low, scale=self._high - self._low)

    def _cdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        return uniform.cdf(x, loc=self._low, scale=self._high - self._low)

    def _ppf_untransformed_untruncated(self, q) -> np.ndarray | float:
        return uniform.ppf(q, loc=self._low, scale=self._high - self._low)


class Laplace(Distribution):
    """A (log-)Laplace distribution.

    :param loc: The location parameter of the distribution.
    :param scale: The scale parameter of the distribution.
    :param trunc: The truncation limits of the distribution.
        ``None`` if the distribution is not truncated.
    :param log: If ``True``, the distribution is transformed to a log-Laplace
        distribution. If a float, the distribution is transformed to a
        log-Laplace distribution with the given base.
        If ``False``, no transformation is applied.

        .. note::

           If a transformation is applied, the location and scale parameters
           and the truncation limits are the location, scale and truncation
           limits of the underlying Laplace distribution.
    """

    def __init__(
        self,
        loc: float,
        scale: float,
        trunc: tuple[float, float] | None = None,
        log: bool | float = False,
    ):
        self._loc = loc
        self._scale = scale
        super().__init__(log=log, trunc=trunc)

    def __repr__(self):
        trunc = f", trunc={self._trunc}" if self._trunc else ""
        log = f", log={self._logbase}" if self._logbase else ""
        return f"Laplace(loc={self._loc}, scale={self._scale}{trunc}{log})"

    def _sample(self, shape=None) -> np.ndarray | float:
        return np.random.laplace(loc=self._loc, scale=self._scale, size=shape)

    def _pdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        return laplace.pdf(x, loc=self._loc, scale=self._scale)

    def _cdf_untransformed_untruncated(self, x) -> np.ndarray | float:
        return laplace.cdf(x, loc=self._loc, scale=self._scale)

    def _ppf_untransformed_untruncated(self, q) -> np.ndarray | float:
        return laplace.ppf(q, loc=self._loc, scale=self._scale)

    @property
    def loc(self) -> float:
        """The location parameter of the underlying distribution."""
        return self._loc

    @property
    def scale(self) -> float:
        """The scale parameter of the underlying distribution."""
        return self._scale
