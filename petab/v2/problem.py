"""PEtab v2 problems."""
from __future__ import annotations

import logging
import os
import tempfile
from math import nan
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ..v1 import (
    conditions,
    core,
    mapping,
    measurements,
    observables,
    parameter_mapping,
    parameters,
    sampling,
    yaml,
)
from ..v1.C import *  # noqa: F403
from ..v1.models.model import Model, model_factory
from ..v1.yaml import get_path_prefix

if TYPE_CHECKING:
    from ..v2.lint import ValidationIssue, ValidationResultList, ValidationTask


__all__ = ["Problem"]


class Problem:
    """
    PEtab parameter estimation problem as defined by

    - model
    - condition table
    - measurement table
    - parameter table
    - observables table
    - mapping table

    Optionally it may contain visualization tables.

    Parameters:
        condition_df: PEtab condition table
        measurement_df: PEtab measurement table
        parameter_df: PEtab parameter table
        observable_df: PEtab observable table
        visualization_df: PEtab visualization table
        mapping_df: PEtab mapping table
        model: The underlying model
        extensions_config: Information on the extensions used
    """

    def __init__(
        self,
        model: Model = None,
        condition_df: pd.DataFrame = None,
        measurement_df: pd.DataFrame = None,
        parameter_df: pd.DataFrame = None,
        visualization_df: pd.DataFrame = None,
        observable_df: pd.DataFrame = None,
        mapping_df: pd.DataFrame = None,
        extensions_config: dict = None,
    ):
        from ..v2.lint import default_validation_tasks

        self.condition_df: pd.DataFrame | None = condition_df
        self.measurement_df: pd.DataFrame | None = measurement_df
        self.parameter_df: pd.DataFrame | None = parameter_df
        self.visualization_df: pd.DataFrame | None = visualization_df
        self.observable_df: pd.DataFrame | None = observable_df
        self.mapping_df: pd.DataFrame | None = mapping_df
        self.model: Model | None = model
        self.extensions_config = extensions_config or {}
        self.validation_tasks: list[
            ValidationTask
        ] = default_validation_tasks.copy()

    def __str__(self):
        model = f"with model ({self.model})" if self.model else "without model"
        conditions = (
            f"{self.condition_df.shape[0]} conditions"
            if self.condition_df is not None
            else "without conditions table"
        )

        observables = (
            f"{self.observable_df.shape[0]} observables"
            if self.observable_df is not None
            else "without observables table"
        )

        measurements = (
            f"{self.measurement_df.shape[0]} measurements"
            if self.measurement_df is not None
            else "without measurements table"
        )

        if self.parameter_df is not None:
            num_estimated_parameters = (
                sum(self.parameter_df[ESTIMATE] == 1)
                if ESTIMATE in self.parameter_df
                else self.parameter_df.shape[0]
            )
            parameters = f"{num_estimated_parameters} estimated parameters"
        else:
            parameters = "without parameter_df table"

        return (
            f"PEtab Problem {model}, {conditions}, {observables}, "
            f"{measurements}, {parameters}"
        )

    @staticmethod
    def from_yaml(yaml_config: dict | Path | str) -> Problem:
        """
        Factory method to load model and tables as specified by YAML file.

        Arguments:
            yaml_config: PEtab configuration as dictionary or YAML file name
        """
        if isinstance(yaml_config, Path):
            yaml_config = str(yaml_config)

        if isinstance(yaml_config, str):
            yaml_file = yaml_config
            path_prefix = get_path_prefix(yaml_file)
            yaml_config = yaml.load_yaml(yaml_config)
            get_path = lambda filename: f"{path_prefix}/{filename}"  # noqa: E731
        else:
            yaml_file = None
            get_path = lambda filename: filename  # noqa: E731

        if yaml_config[FORMAT_VERSION] not in {"2.0.0"}:
            # If we got a path to a v1 yaml file, try to auto-upgrade
            from tempfile import TemporaryDirectory

            from ..versions import get_major_version
            from .petab1to2 import petab1to2

            if get_major_version(yaml_config) == 1 and yaml_file:
                logging.debug(
                    "Auto-upgrading problem from PEtab 1.0 to PEtab 2.0"
                )
                with TemporaryDirectory() as tmpdirname:
                    try:
                        petab1to2(yaml_file, output_dir=tmpdirname)
                    except Exception as e:
                        raise ValueError(
                            "Failed to auto-upgrade PEtab 1.0 problem to "
                            "PEtab 2.0"
                        ) from e
                    return Problem.from_yaml(
                        Path(tmpdirname) / Path(yaml_file).name
                    )
            raise ValueError(
                "Provided PEtab files are of unsupported version "
                f"{yaml_config[FORMAT_VERSION]}. Expected 2.0.0."
            )

        if yaml.is_composite_problem(yaml_config):
            raise ValueError(
                "petab.Problem.from_yaml() can only be used for "
                "yaml files comprising a single model. "
                "Consider using "
                "petab.CompositeProblem.from_yaml() instead."
            )

        problem0 = yaml_config["problems"][0]

        if isinstance(yaml_config[PARAMETER_FILE], list):
            parameter_df = parameters.get_parameter_df(
                [get_path(f) for f in yaml_config[PARAMETER_FILE]]
            )
        else:
            parameter_df = (
                parameters.get_parameter_df(
                    get_path(yaml_config[PARAMETER_FILE])
                )
                if yaml_config[PARAMETER_FILE]
                else None
            )

        if len(problem0[MODEL_FILES]) > 1:
            # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
            raise NotImplementedError(
                "Support for multiple models is not yet implemented."
            )
        if not problem0[MODEL_FILES]:
            model = None
        else:
            model_id, model_info = next(iter(problem0[MODEL_FILES].items()))
            model = model_factory(
                get_path(model_info[MODEL_LOCATION]),
                model_info[MODEL_LANGUAGE],
                model_id=model_id,
            )

        measurement_files = [
            get_path(f) for f in problem0.get(MEASUREMENT_FILES, [])
        ]
        # If there are multiple tables, we will merge them
        measurement_df = (
            core.concat_tables(
                measurement_files, measurements.get_measurement_df
            )
            if measurement_files
            else None
        )

        condition_files = [
            get_path(f) for f in problem0.get(CONDITION_FILES, [])
        ]
        # If there are multiple tables, we will merge them
        condition_df = (
            core.concat_tables(condition_files, conditions.get_condition_df)
            if condition_files
            else None
        )

        visualization_files = [
            get_path(f) for f in problem0.get(VISUALIZATION_FILES, [])
        ]
        # If there are multiple tables, we will merge them
        visualization_df = (
            core.concat_tables(visualization_files, core.get_visualization_df)
            if visualization_files
            else None
        )

        observable_files = [
            get_path(f) for f in problem0.get(OBSERVABLE_FILES, [])
        ]
        # If there are multiple tables, we will merge them
        observable_df = (
            core.concat_tables(observable_files, observables.get_observable_df)
            if observable_files
            else None
        )

        mapping_files = [get_path(f) for f in problem0.get(MAPPING_FILES, [])]
        # If there are multiple tables, we will merge them
        mapping_df = (
            core.concat_tables(mapping_files, mapping.get_mapping_df)
            if mapping_files
            else None
        )

        return Problem(
            condition_df=condition_df,
            measurement_df=measurement_df,
            parameter_df=parameter_df,
            observable_df=observable_df,
            model=model,
            visualization_df=visualization_df,
            mapping_df=mapping_df,
            extensions_config=yaml_config.get(EXTENSIONS, {}),
        )

    @staticmethod
    def from_combine(filename: Path | str) -> Problem:
        """Read PEtab COMBINE archive (http://co.mbine.org/documents/archive).

        See also :py:func:`petab.create_combine_archive`.

        Arguments:
            filename: Path to the PEtab-COMBINE archive

        Returns:
            A :py:class:`petab.Problem` instance.
        """
        # function-level import, because module-level import interfered with
        # other SWIG interfaces
        try:
            import libcombine
        except ImportError as e:
            raise ImportError(
                "To use PEtab's COMBINE functionality, libcombine "
                "(python-libcombine) must be installed."
            ) from e

        archive = libcombine.CombineArchive()
        if archive.initializeFromArchive(str(filename)) is None:
            raise ValueError(f"Invalid Combine Archive: {filename}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            archive.extractTo(tmpdirname)
            problem = Problem.from_yaml(
                os.path.join(tmpdirname, archive.getMasterFile().getLocation())
            )
        archive.cleanUp()

        return problem

    @staticmethod
    def get_problem(problem: str | Path | Problem) -> Problem:
        """Get a PEtab problem from a file or a problem object.

        Arguments:
            problem: Path to a PEtab problem file or a PEtab problem object.

        Returns:
            A PEtab problem object.
        """
        if isinstance(problem, Problem):
            return problem

        if isinstance(problem, str | Path):
            return Problem.from_yaml(problem)

        raise TypeError(
            "The argument `problem` must be a path to a PEtab problem file "
            "or a PEtab problem object."
        )

    def get_optimization_parameters(self) -> list[str]:
        """
        Return list of optimization parameter IDs.

        See :py:func:`petab.parameters.get_optimization_parameters`.
        """
        return parameters.get_optimization_parameters(self.parameter_df)

    def get_optimization_parameter_scales(self) -> dict[str, str]:
        """
        Return list of optimization parameter scaling strings.

        See :py:func:`petab.parameters.get_optimization_parameters`.
        """
        return parameters.get_optimization_parameter_scaling(self.parameter_df)

    def get_observable_ids(self) -> list[str]:
        """
        Returns dictionary of observable ids.
        """
        return list(self.observable_df.index)

    def _apply_mask(self, v: list, free: bool = True, fixed: bool = True):
        """Apply mask of only free or only fixed values.

        Parameters
        ----------
        v:
            The full vector the mask is to be applied to.
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.

        Returns
        -------
        The reduced vector with applied mask.
        """
        if not free and not fixed:
            return []
        if not free:
            return [v[ix] for ix in self.x_fixed_indices]
        if not fixed:
            return [v[ix] for ix in self.x_free_indices]
        return v

    def get_x_ids(self, free: bool = True, fixed: bool = True):
        """Generic function to get parameter ids.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.

        Returns
        -------
        The parameter IDs.
        """
        v = list(self.parameter_df.index.values)
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def x_ids(self) -> list[str]:
        """Parameter table parameter IDs"""
        return self.get_x_ids()

    @property
    def x_free_ids(self) -> list[str]:
        """Parameter table parameter IDs, for free parameters."""
        return self.get_x_ids(fixed=False)

    @property
    def x_fixed_ids(self) -> list[str]:
        """Parameter table parameter IDs, for fixed parameters."""
        return self.get_x_ids(free=False)

    def get_x_nominal(
        self, free: bool = True, fixed: bool = True, scaled: bool = False
    ):
        """Generic function to get parameter nominal values.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.
        scaled:
            Whether to scale the values according to the parameter scale,
            or return them on linear scale.

        Returns
        -------
        The parameter nominal values.
        """
        if NOMINAL_VALUE in self.parameter_df:
            v = list(self.parameter_df[NOMINAL_VALUE])
        else:
            v = [nan] * len(self.parameter_df)

        if scaled:
            v = list(
                parameters.map_scale(v, self.parameter_df[PARAMETER_SCALE])
            )
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def x_nominal(self) -> list:
        """Parameter table nominal values"""
        return self.get_x_nominal()

    @property
    def x_nominal_free(self) -> list:
        """Parameter table nominal values, for free parameters."""
        return self.get_x_nominal(fixed=False)

    @property
    def x_nominal_fixed(self) -> list:
        """Parameter table nominal values, for fixed parameters."""
        return self.get_x_nominal(free=False)

    @property
    def x_nominal_scaled(self) -> list:
        """Parameter table nominal values with applied parameter scaling"""
        return self.get_x_nominal(scaled=True)

    @property
    def x_nominal_free_scaled(self) -> list:
        """Parameter table nominal values with applied parameter scaling,
        for free parameters.
        """
        return self.get_x_nominal(fixed=False, scaled=True)

    @property
    def x_nominal_fixed_scaled(self) -> list:
        """Parameter table nominal values with applied parameter scaling,
        for fixed parameters.
        """
        return self.get_x_nominal(free=False, scaled=True)

    def get_lb(
        self, free: bool = True, fixed: bool = True, scaled: bool = False
    ):
        """Generic function to get lower parameter bounds.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.
        scaled:
            Whether to scale the values according to the parameter scale,
            or return them on linear scale.

        Returns
        -------
        The lower parameter bounds.
        """
        v = list(self.parameter_df[LOWER_BOUND])
        if scaled:
            v = list(
                parameters.map_scale(v, self.parameter_df[PARAMETER_SCALE])
            )
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def lb(self) -> list:
        """Parameter table lower bounds."""
        return self.get_lb()

    @property
    def lb_scaled(self) -> list:
        """Parameter table lower bounds with applied parameter scaling"""
        return self.get_lb(scaled=True)

    def get_ub(
        self, free: bool = True, fixed: bool = True, scaled: bool = False
    ):
        """Generic function to get upper parameter bounds.

        Parameters
        ----------
        free:
            Whether to return free parameters, i.e. parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e. parameters not to
            estimate.
        scaled:
            Whether to scale the values according to the parameter scale,
            or return them on linear scale.

        Returns
        -------
        The upper parameter bounds.
        """
        v = list(self.parameter_df[UPPER_BOUND])
        if scaled:
            v = list(
                parameters.map_scale(v, self.parameter_df[PARAMETER_SCALE])
            )
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def ub(self) -> list:
        """Parameter table upper bounds"""
        return self.get_ub()

    @property
    def ub_scaled(self) -> list:
        """Parameter table upper bounds with applied parameter scaling"""
        return self.get_ub(scaled=True)

    @property
    def x_free_indices(self) -> list[int]:
        """Parameter table estimated parameter indices."""
        estimated = list(self.parameter_df[ESTIMATE])
        return [j for j, val in enumerate(estimated) if val != 0]

    @property
    def x_fixed_indices(self) -> list[int]:
        """Parameter table non-estimated parameter indices."""
        estimated = list(self.parameter_df[ESTIMATE])
        return [j for j, val in enumerate(estimated) if val == 0]

    def get_simulation_conditions_from_measurement_df(self) -> pd.DataFrame:
        """See :func:`petab.get_simulation_conditions`."""
        return measurements.get_simulation_conditions(self.measurement_df)

    def get_optimization_to_simulation_parameter_mapping(self, **kwargs):
        """
        See
        :py:func:`petab.parameter_mapping.get_optimization_to_simulation_parameter_mapping`,
        to which all keyword arguments are forwarded.
        """
        return (
            parameter_mapping.get_optimization_to_simulation_parameter_mapping(
                condition_df=self.condition_df,
                measurement_df=self.measurement_df,
                parameter_df=self.parameter_df,
                observable_df=self.observable_df,
                model=self.model,
                **kwargs,
            )
        )

    def create_parameter_df(self, **kwargs) -> pd.DataFrame:
        """Create a new PEtab parameter table

        See :py:func:`create_parameter_df`.
        """
        return parameters.create_parameter_df(
            model=self.model,
            condition_df=self.condition_df,
            observable_df=self.observable_df,
            measurement_df=self.measurement_df,
            mapping_df=self.mapping_df,
            **kwargs,
        )

    def sample_parameter_startpoints(self, n_starts: int = 100, **kwargs):
        """Create 2D array with starting points for optimization

        See :py:func:`petab.sample_parameter_startpoints`.
        """
        return sampling.sample_parameter_startpoints(
            self.parameter_df, n_starts=n_starts, **kwargs
        )

    def sample_parameter_startpoints_dict(
        self, n_starts: int = 100
    ) -> list[dict[str, float]]:
        """Create dictionaries with starting points for optimization

        See also :py:func:`petab.sample_parameter_startpoints`.

        Returns:
            A list of dictionaries with parameter IDs mapping to samples
            parameter values.
        """
        return [
            dict(zip(self.x_free_ids, parameter_values, strict=True))
            for parameter_values in self.sample_parameter_startpoints(
                n_starts=n_starts
            )
        ]

    def unscale_parameters(
        self,
        x_dict: dict[str, float],
    ) -> dict[str, float]:
        """Unscale parameter values.

        Parameters
        ----------
        x_dict:
            Keys are parameter IDs in the PEtab problem, values are scaled
            parameter values.

        Returns
        -------
        The unscaled parameter values.
        """
        return {
            parameter_id: parameters.unscale(
                parameter_value,
                self.parameter_df[PARAMETER_SCALE][parameter_id],
            )
            for parameter_id, parameter_value in x_dict.items()
        }

    def scale_parameters(
        self,
        x_dict: dict[str, float],
    ) -> dict[str, float]:
        """Scale parameter values.

        Parameters
        ----------
        x_dict:
            Keys are parameter IDs in the PEtab problem, values are unscaled
            parameter values.

        Returns
        -------
        The scaled parameter values.
        """
        return {
            parameter_id: parameters.scale(
                parameter_value,
                self.parameter_df[PARAMETER_SCALE][parameter_id],
            )
            for parameter_id, parameter_value in x_dict.items()
        }

    @property
    def n_estimated(self) -> int:
        """The number of estimated parameters."""
        return len(self.x_free_indices)

    @property
    def n_measurements(self) -> int:
        """Number of measurements."""
        return self.measurement_df[MEASUREMENT].notna().sum()

    @property
    def n_priors(self) -> int:
        """Number of priors."""
        if OBJECTIVE_PRIOR_PARAMETERS not in self.parameter_df:
            return 0

        return self.parameter_df[OBJECTIVE_PRIOR_PARAMETERS].notna().sum()

    def validate(
        self, validation_tasks: list[ValidationTask] = None
    ) -> ValidationResultList:
        """Validate the PEtab problem.

        Arguments:
            validation_tasks: List of validation tasks to run. If ``None``
             or empty, :attr:`Problem.validation_tasks` are used.
        Returns:
            A list of validation results.
        """
        from ..v2.lint import ValidationIssueSeverity, ValidationResultList

        validation_results = ValidationResultList()
        if self.extensions_config:
            validation_results.append(
                ValidationIssue(
                    ValidationIssueSeverity.WARNING,
                    "Validation of PEtab extensions is not yet implemented, "
                    "but the given problem uses the following extensions: "
                    f"{'', ''.join(self.extensions_config.keys())}",
                )
            )

        for task in validation_tasks or self.validation_tasks:
            try:
                cur_result = task.run(self)
            except Exception as e:
                cur_result = ValidationIssue(
                    ValidationIssueSeverity.CRITICAL,
                    f"Validation task {task} failed with exception: {e}",
                )

            if cur_result:
                validation_results.append(cur_result)

                if cur_result.level == ValidationIssueSeverity.CRITICAL:
                    break

        return validation_results
