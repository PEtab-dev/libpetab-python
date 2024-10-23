"""Probability distributions used by PEtab."""
from __future__ import annotations

import abc
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import laplace, lognorm, norm, uniform

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

    #: Mapping from distribution type to distribution class for factory method.
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

        self.type = type_
        self.parameters = parameters
        self.bounds = bounds
        self.transformation = transformation
        self.parameter_scale = parameter_scale

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.parameters[0]!r}, {self.parameters[1]!r},"
            f" bounds={self.bounds!r}, transformation={self.transformation!r}"
            ")"
        )

    @abc.abstractmethod
    def sample(self, shape=None) -> np.ndarray:
        """Sample from the distribution.

        :param shape: The shape of the sample.
        :return: A sample from the distribution.
        """
        ...

    def _scale_sample(self, sample):
        """Scale the sample to the parameter space"""
        if self.parameter_scale:
            return sample

        return scale(sample, self.transformation)

    def _clip_to_bounds(self, x):
        """Clip `x` values to bounds.

        :param x: The values to clip. Assumed to be on the parameter scale.
        """
        # TODO: replace this by proper truncation
        if self.bounds is None:
            return x

        return np.maximum(
            np.minimum(self.ub_scaled, x),
            self.lb_scaled,
        )

    @property
    def lb_scaled(self):
        """The lower bound on the parameter scale."""
        return scale(self.bounds[0], self.transformation)

    @property
    def ub_scaled(self):
        """The upper bound on the parameter scale."""
        return scale(self.bounds[1], self.transformation)

    @abc.abstractmethod
    def _pdf(self, x):
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
            ``x`` is assumed to be on the linear scale.
        :return: The value of the PDF at ``x``.
        """
        ...

    def pdf(self, x):
        """Probability density function at x.

        :param x: The value at which to evaluate the PDF.
            ``x`` is assumed to be on the parameter scale.
        :return: The value of the PDF at ``x``. Note that the PDF does
            currently not account for the clipping at the bounds.
        """
        x = x if self.parameter_scale else unscale(x, self.transformation)

        # scale the PDF to the parameter scale
        if self.parameter_scale or self.transformation == C.LIN:
            coeff = 1
        elif self.transformation == C.LOG10:
            coeff = x * np.log(10)
        elif self.transformation == C.LOG:
            coeff = x
        else:
            raise ValueError(f"Unknown transformation: {self.transformation}")

        return self._pdf(x) * coeff

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
        sample = np.random.normal(
            loc=self.parameters[0], scale=self.parameters[1], size=shape
        )
        return self._clip_to_bounds(self._scale_sample(sample))

    def _pdf(self, x):
        return norm.pdf(x, loc=self.parameters[0], scale=self.parameters[1])


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
        sample = np.random.lognormal(
            mean=self.parameters[0], sigma=self.parameters[1], size=shape
        )
        return self._clip_to_bounds(self._scale_sample(sample))

    def _pdf(self, x):
        return lognorm.pdf(
            x, scale=np.exp(self.parameters[0]), s=self.parameters[1]
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

    def sample(self, shape=None):
        sample = np.random.uniform(
            low=self.parameters[0], high=self.parameters[1], size=shape
        )
        return self._clip_to_bounds(self._scale_sample(sample))

    def _pdf(self, x):
        return uniform.pdf(
            x,
            loc=self.parameters[0],
            scale=self.parameters[1] - self.parameters[0],
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
        sample = np.random.laplace(
            loc=self.parameters[0], scale=self.parameters[1], size=shape
        )
        return self._clip_to_bounds(self._scale_sample(sample))

    def _pdf(self, x):
        return laplace.pdf(x, loc=self.parameters[0], scale=self.parameters[1])


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
        return self.parameters[0]

    @property
    def scale(self):
        """The scale of the underlying Laplace distribution."""
        return self.parameters[1]

    def sample(self, shape=None):
        sample = np.exp(
            np.random.laplace(loc=self.mean, scale=self.scale, size=shape)
        )
        return self._clip_to_bounds(self._scale_sample(sample))

    def _pdf(self, x):
        return (
            1
            / (2 * self.scale * x)
            * np.exp(-np.abs(np.log(x) - self.mean) / self.scale)
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
            bounds=bounds,
            mean=mean,
            std=std,
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
