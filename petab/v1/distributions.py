"""Probability distributions used by PEtab."""
from __future__ import annotations

import abc
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import laplace, lognorm, norm, truncnorm, uniform

from . import C
from .parameters import scale, unscale

__all__ = [
    "Distribution",
    "Normal",
    "LogNormal",
    "Uniform",
    "Laplace",
    "LogLaplace",
    "ParameterScaleNormal",
    "ParameterScaleUniform",
    "ParameterScaleLaplace",
]


class Distribution(abc.ABC):
    """A univariate probability distribution.

    :param type_: The type of the distribution.
    :param transformation: The transformation to be applied to the sample.
        Ignored if `parameter_scale` is `True`.
    :param parameters: The parameters of the distribution (unaffected by
        `parameter_scale` and `transformation`).
    :param bounds: The untransformed bounds of the sample (lower, upper).
    :param parameter_scale: Whether the parameters are already on the correct
        scale. If `False`, the parameters are transformed to the correct scale.
        If `True`, the parameters are assumed to be on the correct scale and
        no transformation is applied.
    :param transformation: The transformation of the distribution.
    """

    _type_to_cls: dict[str, type[Distribution]] = {}

    def __init__(
        self,
        type_: str,
        transformation: str,
        parameters: tuple,
        bounds: tuple = None,
        parameter_scale: bool = False,
    ):
        if type_ not in self._type_to_cls:
            raise ValueError(f"Unknown distribution type: {type_}")

        if len(parameters) != 2:
            raise ValueError(
                f"Expected two parameters, got {len(parameters)}: {parameters}"
            )

        if bounds is not None and len(bounds) != 2:
            raise ValueError(
                f"Expected two bounds, got {len(bounds)}: {bounds}"
            )

        self._type = type_
        self._parameters = parameters
        self._bounds = bounds
        self._transformation = transformation
        self._parameter_scale = parameter_scale
        # normalization factor for pdf/cdf of truncated distributions
        self._truncation_normalizer = 1
        # lower and upper bounds on the same scale as the `distribution`
        #  parameters (not necessarily the PEtab parameter scale)
        self._low = None
        self._high = None

        if self._bounds is not None:
            self._low = (
                self.lb_scaled if self._parameter_scale else self._bounds[0]
            )
            self._high = (
                self.ub_scaled if self._parameter_scale else self._bounds[1]
            )
            try:
                self._cd_low = self._cdf_unscaled_untruncated(self._low)
                self._cd_high = self._cdf_unscaled_untruncated(self._high)
                self._truncation_normalizer = 1 / (
                    self._cd_high - self._cd_low
                )
            except NotImplementedError:
                pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self._parameters[0]!r}, {self._parameters[1]!r},"
            f" bounds={self._bounds!r}, "
            f"transformation={self._transformation!r}"
            ")"
        )

    @property
    def bounds(self):
        """The bounds of the distribution."""
        return self._bounds

    @abc.abstractmethod
    def sample(self, shape=None) -> np.ndarray:
        """Sample from the distribution.

        :param shape: The shape of the sample.
        :return: A sample from the distribution.
        """
        ...

    def _scale_sample(self, sample):
        """Scale the sample to the parameter space"""
        if self._parameter_scale:
            return sample

        return scale(sample, self._transformation)

    @property
    def lb_scaled(self):
        """The lower bound on the parameter scale."""
        return scale(self._bounds[0], self._transformation)

    @property
    def ub_scaled(self):
        """The upper bound on the parameter scale."""
        return scale(self._bounds[1], self._transformation)

    @abc.abstractmethod
    def _pdf(self, x):
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
            ``x`` is assumed to be on the linear scale.
        :return: The value of the PDF at ``x``.
        """
        ...

    def _cdf(self, x):
        """Cumulative distribution function at x.

        :param x: The value at which to evaluate the CDF.
            ``x`` is assumed to be on the linear scale.
        :return: The value of the CDF at ``x``.
        """
        raise NotImplementedError

    def _cdf_unscaled_untruncated(self, x):
        """Cumulative distribution function at x, ignoring scale and bounds.

        :param x: The value at which to evaluate the CDF.
            ``x`` is assumed to be on the parameter scale.
        :return: The value of the CDF at ``x``.
        """
        raise NotImplementedError

    def pdf(self, x):
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
            ``x`` is assumed to be on the parameter scale.
        :return: The value of the PDF at ``x``.
        """
        x = x if self._parameter_scale else unscale(x, self._transformation)

        # scale the PDF to the parameter scale
        if self._parameter_scale or self._transformation == C.LIN:
            coeff = 1
        elif self._transformation == C.LOG10:
            coeff = x * np.log(10)
        elif self._transformation == C.LOG:
            coeff = x
        else:
            raise ValueError(f"Unknown transformation: {self._transformation}")

        return self._pdf(x) * coeff

    def _ppf_unscaled_untruncated(self, q):
        """Percent point function at q, ignoring scale and bounds.

        :param q: The quantile at which to evaluate the PPF.
        :return: The value of the PPF at ``q``.
        """
        raise NotImplementedError

    def _inverse_transform_sample(self, shape):
        """Generate an inverse transform sample for the unscaled,
        untruncated distribution.

        :param shape: The shape of the sample.
        :return: The sample.
        """
        uniform_sample = np.random.uniform(
            low=self._cd_low, high=self._cd_high, size=shape
        )
        return self._ppf_unscaled_untruncated(uniform_sample)

    def neglogprior(self, x):
        """Negative log-prior at x.

        :param x: The value at which to evaluate the negative log-prior.
            ``x`` is assumed to be on the parameter scale.
        :return: The negative log-prior at ``x``.
        """
        return -np.log(self.pdf(x))

    @staticmethod
    def from_par_dict(
        d, type_=Literal["initialization", "objective"]
    ) -> Distribution:
        """Create a distribution from a row of the parameter table.

        :param d: A dictionary representing a row of the parameter table.
        :param type_: The type of the distribution.
        :return: A distribution object.
        """
        dist_type = d.get(f"{type_}PriorType", C.NORMAL)
        if not isinstance(dist_type, str) and np.isnan(dist_type):
            dist_type = C.PARAMETER_SCALE_UNIFORM
        cls = Distribution._type_to_cls[dist_type]

        pscale = d.get(C.PARAMETER_SCALE, C.LIN)
        if (
            pd.isna(d[f"{type_}PriorParameters"])
            and dist_type == C.PARAMETER_SCALE_UNIFORM
        ):
            params = (
                scale(d[C.LOWER_BOUND], pscale),
                scale(d[C.UPPER_BOUND], pscale),
            )
        else:
            params = tuple(
                map(
                    float,
                    d[f"{type_}PriorParameters"].split(C.PARAMETER_SEPARATOR),
                )
            )
        return cls(
            *params,
            bounds=(d[C.LOWER_BOUND], d[C.UPPER_BOUND]),
            transformation=pscale,
        )


class Normal(Distribution):
    """A normal distribution."""

    def __init__(
        self,
        mean: float,
        std: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
        _type=C.NORMAL,
        _parameter_scale=False,
    ):
        super().__init__(
            transformation=transformation,
            parameters=(mean, std),
            bounds=bounds,
            parameter_scale=_parameter_scale,
            type_=_type,
        )

    def sample(self, shape=None):
        if self._bounds is None:
            sample = np.random.normal(
                loc=self._parameters[0], scale=self._parameters[1], size=shape
            )
        else:
            sample = truncnorm.rvs(
                a=(self._low - self._parameters[0]) / self._parameters[1],
                b=(self._high - self._parameters[0]) / self._parameters[1],
                loc=self._parameters[0],
                scale=self._parameters[1],
                size=shape,
            )
        return self._scale_sample(sample)

    def _pdf(self, x):
        if self._bounds is None:
            return norm.pdf(
                x, loc=self._parameters[0], scale=self._parameters[1]
            )

        return truncnorm.pdf(
            x,
            a=(self._low - self._parameters[0]) / self._parameters[1],
            b=(self._high - self._parameters[0]) / self._parameters[1],
            loc=self._parameters[0],
            scale=self._parameters[1],
        )


class LogNormal(Distribution):
    """A log-normal distribution."""

    def __init__(
        self,
        mean: float,
        std: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
    ):
        super().__init__(
            C.LOG_NORMAL,
            transformation=transformation,
            parameters=(mean, std),
            bounds=bounds,
            parameter_scale=False,
        )

    def sample(self, shape=None):
        if self._bounds is None:
            sample = np.random.lognormal(
                mean=self._parameters[0], sigma=self._parameters[1], size=shape
            )
        else:
            sample = self._inverse_transform_sample(shape)
        return self._scale_sample(sample)

    def _pdf(self, x):
        return (
            lognorm.pdf(
                x, scale=np.exp(self._parameters[0]), s=self._parameters[1]
            )
            * self._truncation_normalizer
        )

    def _cdf_unscaled_untruncated(self, x):
        return lognorm.cdf(
            x, scale=np.exp(self._parameters[0]), s=self._parameters[1]
        )

    def _ppf_unscaled_untruncated(self, q):
        return lognorm.ppf(
            q, scale=np.exp(self._parameters[0]), s=self._parameters[1]
        )


class Uniform(Distribution):
    """A uniform distribution."""

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
        _type=C.UNIFORM,
        _parameter_scale=False,
    ):
        super().__init__(
            _type,
            transformation=transformation,
            parameters=(lower_bound, upper_bound),
            bounds=bounds,
            parameter_scale=_parameter_scale,
        )

    def _low_high(self):
        """Get the lower and upper bounds of the distribution.

        Consider the bounds of the distribution and the parameter bounds.
        The values are not scaled, unless `parameter_scale` is `True`.
        """
        if self._bounds is None:
            return self._parameters

        low = max(
            self._parameters[0],
            self.lb_scaled if self._parameter_scale else self._bounds[0],
        )
        high = min(
            self._parameters[1],
            self.ub_scaled if self._parameter_scale else self._bounds[1],
        )

        return low, high

    def sample(self, shape=None):
        low, high = self._low_high()
        sample = np.random.uniform(low=low, high=high, size=shape)
        return self._scale_sample(sample)

    def _pdf(self, x):
        low, high = self._low_high()
        return uniform.pdf(
            x,
            loc=low,
            scale=high - low,
        )


class Laplace(Distribution):
    """A Laplace distribution."""

    def __init__(
        self,
        mean: float,
        scale: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
        _type=C.LAPLACE,
        _parameter_scale=False,
    ):
        super().__init__(
            _type,
            transformation=transformation,
            parameters=(mean, scale),
            bounds=bounds,
            parameter_scale=_parameter_scale,
        )

    def sample(self, shape=None):
        if self._bounds is None:
            sample = np.random.laplace(
                loc=self._parameters[0], scale=self._parameters[1], size=shape
            )
        else:
            sample = self._inverse_transform_sample(shape)
        return self._scale_sample(sample)

    def _pdf(self, x):
        return (
            laplace.pdf(x, loc=self._parameters[0], scale=self._parameters[1])
            * self._truncation_normalizer
        )

    def _cdf_unscaled_untruncated(self, x):
        return laplace.cdf(
            x, loc=self._parameters[0], scale=self._parameters[1]
        )

    def _cdf_unscaled_truncated(self, x):
        if self._bounds is None:
            return self._cdf_unscaled_untruncated(x)

        cd_untruncated = self._cdf_unscaled_untruncated(x)
        return (cd_untruncated - self._cd_low) * self._truncation_normalizer

    def _ppf_unscaled_untruncated(self, q):
        return laplace.ppf(
            q, loc=self._parameters[0], scale=self._parameters[1]
        )


class LogLaplace(Distribution):
    """A log-Laplace distribution."""

    def __init__(
        self,
        mean: float,
        scale: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
    ):
        super().__init__(
            C.LOG_LAPLACE,
            transformation=transformation,
            parameters=(mean, scale),
            bounds=bounds,
            parameter_scale=False,
        )

    @property
    def mean(self):
        """The mean of the underlying Laplace distribution."""
        return self._parameters[0]

    @property
    def scale(self):
        """The scale of the underlying Laplace distribution."""
        return self._parameters[1]

    def sample(self, shape=None):
        if self._bounds is None:
            sample = np.exp(
                np.random.laplace(loc=self.mean, scale=self.scale, size=shape)
            )
        else:
            sample = self._inverse_transform_sample(shape)

        return self._scale_sample(sample)

    def _pdf(self, x):
        return (
            1
            / (2 * self.scale * x)
            * np.exp(-np.abs(np.log(x) - self.mean) / self.scale)
        ) * self._truncation_normalizer

    def _cdf_unscaled_untruncated(self, x):
        return laplace.cdf(
            np.log(x), loc=self._parameters[0], scale=self._parameters[1]
        )

    def _ppf_unscaled_untruncated(self, q):
        return np.exp(
            laplace.ppf(q, loc=self._parameters[0], scale=self._parameters[1])
        )


class ParameterScaleNormal(Normal):
    """A normal distribution with parameters on the parameter scale."""

    def __init__(
        self,
        mean: float,
        std: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
    ):
        super().__init__(
            transformation=transformation,
            mean=mean,
            std=std,
            bounds=bounds,
            _type=C.PARAMETER_SCALE_NORMAL,
            _parameter_scale=True,
        )


class ParameterScaleUniform(Uniform):
    """A uniform distribution with parameters on the parameter scale."""

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
    ):
        super().__init__(
            transformation=transformation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bounds=bounds,
            _type=C.PARAMETER_SCALE_UNIFORM,
            _parameter_scale=True,
        )


class ParameterScaleLaplace(Laplace):
    """A Laplace distribution with parameters on the parameter scale."""

    def __init__(
        self,
        mean: float,
        scale: float,
        bounds: tuple = None,
        transformation: str = C.LIN,
    ):
        super().__init__(
            _type=C.PARAMETER_SCALE_LAPLACE,
            transformation=transformation,
            mean=mean,
            scale=scale,
            bounds=bounds,
            _parameter_scale=True,
        )


Distribution._type_to_cls = {
    C.NORMAL: Normal,
    C.LOG_NORMAL: LogNormal,
    C.UNIFORM: Uniform,
    C.LAPLACE: Laplace,
    C.LOG_LAPLACE: LogLaplace,
    C.PARAMETER_SCALE_NORMAL: ParameterScaleNormal,
    C.PARAMETER_SCALE_UNIFORM: ParameterScaleUniform,
    C.PARAMETER_SCALE_LAPLACE: ParameterScaleLaplace,
}
