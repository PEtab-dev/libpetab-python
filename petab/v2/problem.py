"""PEtab v2 problems."""

from __future__ import annotations

import logging
import os
import tempfile
import traceback
from collections.abc import Sequence
from math import nan
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import sympy as sp
from pydantic import AnyUrl, BaseModel, Field, field_validator

from ..v1 import (
    mapping,
    measurements,
    observables,
    parameter_mapping,
    parameters,
    validate_yaml_syntax,
    yaml,
)
from ..v1.core import concat_tables, get_visualization_df
from ..v1.distributions import Distribution
from ..v1.models.model import Model, model_factory
from ..v1.yaml import get_path_prefix
from ..v2.C import *  # noqa: F403
from ..versions import parse_version
from . import conditions, core, experiments

if TYPE_CHECKING:
    from ..v2.lint import ValidationResultList, ValidationTask


__all__ = ["Problem", "ProblemConfig"]


class Problem:
    """
    PEtab parameter estimation problem

    A PEtab parameter estimation problem as defined by

    - model
    - condition table
    - experiment table
    - measurement table
    - parameter table
    - observable table
    - mapping table

    Optionally, it may contain visualization tables.

    See also :doc:`petab:v2/documentation_data_format`.
    """

    def __init__(
        self,
        model: Model = None,
        condition_table: core.ConditionTable = None,
        experiment_table: core.ExperimentTable = None,
        observable_table: core.ObservableTable = None,
        measurement_table: core.MeasurementTable = None,
        parameter_table: core.ParameterTable = None,
        mapping_table: core.MappingTable = None,
        visualization_df: pd.DataFrame = None,
        config: ProblemConfig = None,
    ):
        from ..v2.lint import default_validation_tasks

        self.config = config
        self.model: Model | None = model
        self.validation_tasks: list[ValidationTask] = (
            default_validation_tasks.copy()
        )

        self.observable_table = observable_table or core.ObservableTable(
            observables=[]
        )
        self.condition_table = condition_table or core.ConditionTable(
            conditions=[]
        )
        self.experiment_table = experiment_table or core.ExperimentTable(
            experiments=[]
        )
        self.measurement_table = measurement_table or core.MeasurementTable(
            measurements=[]
        )
        self.mapping_table = mapping_table or core.MappingTable(mappings=[])
        self.parameter_table = parameter_table or core.ParameterTable(
            parameters=[]
        )

        self.visualization_df = visualization_df

    def __str__(self):
        model = f"with model ({self.model})" if self.model else "without model"

        ne = len(self.experiment_table.experiments)
        experiments = f"{ne} experiments"

        nc = len(self.condition_table.conditions)
        conditions = f"{nc} conditions"

        no = len(self.observable_table.observables)
        observables = f"{no} observables"

        nm = len(self.measurement_table.measurements)
        measurements = f"{nm} measurements"

        nest = self.parameter_table.n_estimated
        parameters = f"{nest} estimated parameters"

        return (
            f"PEtab Problem {model}, {conditions}, {experiments}, "
            f"{observables}, {measurements}, {parameters}"
        )

    def __getitem__(self, key):
        """Get PEtab entity by ID.

        This allows accessing PEtab entities such as conditions, experiments,
        observables, and parameters by their ID.

        Accessing model entities is not currently not supported.
        """
        for table in (
            self.condition_table,
            self.experiment_table,
            self.observable_table,
            self.measurement_table,
            self.parameter_table,
            self.mapping_table,
        ):
            if table is not None:
                try:
                    return table[key]
                except KeyError:
                    pass

        raise KeyError(
            f"Entity with ID '{key}' not found in the PEtab problem"
        )

    @staticmethod
    def from_yaml(
        yaml_config: dict | Path | str, base_path: str | Path = None
    ) -> Problem:
        """
        Factory method to load model and tables as specified by YAML file.

        Arguments:
            yaml_config: PEtab configuration as dictionary or YAML file name
            base_path: Base directory or URL to resolve relative paths
        """
        if isinstance(yaml_config, Path):
            yaml_config = str(yaml_config)

        if isinstance(yaml_config, str):
            yaml_file = yaml_config
            if base_path is None:
                base_path = get_path_prefix(yaml_file)
            yaml_config = yaml.load_yaml(yaml_file)
        else:
            yaml_file = None

        validate_yaml_syntax(yaml_config)

        def get_path(filename):
            if base_path is None:
                return filename
            return f"{base_path}/{filename}"

        if (format_version := parse_version(yaml_config[FORMAT_VERSION]))[
            0
        ] != 2:
            # If we got a path to a v1 yaml file, try to auto-upgrade
            from tempfile import TemporaryDirectory

            from .petab1to2 import petab1to2

            if format_version[0] == 1 and yaml_file:
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
                f"{yaml_config[FORMAT_VERSION]}."
            )

        if len(yaml_config[MODEL_FILES]) > 1:
            raise ValueError(
                "petab.v2.Problem.from_yaml() can only be used for "
                "yaml files comprising a single model. "
                "Consider using "
                "petab.v2.CompositeProblem.from_yaml() instead."
            )
        config = ProblemConfig(
            **yaml_config, base_path=base_path, filepath=yaml_file
        )
        parameter_df = parameters.get_parameter_df(
            [get_path(f) for f in config.parameter_files]
        )

        if len(config.model_files or []) > 1:
            # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
            raise NotImplementedError(
                "Support for multiple models is not yet implemented."
            )
        model = None
        if config.model_files:
            model_id, model_info = next(iter(config.model_files.items()))
            model = model_factory(
                get_path(model_info.location),
                model_info.language,
                model_id=model_id,
            )

        measurement_files = [get_path(f) for f in config.measurement_files]
        # If there are multiple tables, we will merge them
        measurement_df = (
            concat_tables(measurement_files, measurements.get_measurement_df)
            if measurement_files
            else None
        )

        condition_files = [get_path(f) for f in config.condition_files]
        # If there are multiple tables, we will merge them
        condition_df = (
            concat_tables(condition_files, conditions.get_condition_df)
            if condition_files
            else None
        )

        experiment_files = [get_path(f) for f in config.experiment_files]
        # If there are multiple tables, we will merge them
        experiment_df = (
            concat_tables(experiment_files, experiments.get_experiment_df)
            if experiment_files
            else None
        )

        # TODO: remove in v2?!
        visualization_files = [get_path(f) for f in config.visualization_files]
        # If there are multiple tables, we will merge them
        visualization_df = (
            concat_tables(visualization_files, get_visualization_df)
            if visualization_files
            else None
        )

        observable_files = [get_path(f) for f in config.observable_files]
        # If there are multiple tables, we will merge them
        observable_df = (
            concat_tables(observable_files, observables.get_observable_df)
            if observable_files
            else None
        )

        mapping_files = [get_path(f) for f in config.mapping_files]
        # If there are multiple tables, we will merge them
        mapping_df = (
            concat_tables(mapping_files, mapping.get_mapping_df)
            if mapping_files
            else None
        )

        return Problem.from_dfs(
            condition_df=condition_df,
            experiment_df=experiment_df,
            measurement_df=measurement_df,
            parameter_df=parameter_df,
            observable_df=observable_df,
            model=model,
            visualization_df=visualization_df,
            mapping_df=mapping_df,
            config=config,
        )

    @staticmethod
    def from_dfs(
        model: Model = None,
        condition_df: pd.DataFrame = None,
        experiment_df: pd.DataFrame = None,
        measurement_df: pd.DataFrame = None,
        parameter_df: pd.DataFrame = None,
        visualization_df: pd.DataFrame = None,
        observable_df: pd.DataFrame = None,
        mapping_df: pd.DataFrame = None,
        config: ProblemConfig = None,
    ):
        """
        Construct a PEtab problem from dataframes.

        Parameters:
            condition_df: PEtab condition table
            experiment_df: PEtab experiment table
            measurement_df: PEtab measurement table
            parameter_df: PEtab parameter table
            observable_df: PEtab observable table
            visualization_df: PEtab visualization table
            mapping_df: PEtab mapping table
            model: The underlying model
            config: The PEtab problem configuration
        """

        observable_table = core.ObservableTable.from_df(observable_df)
        condition_table = core.ConditionTable.from_df(condition_df)
        experiment_table = core.ExperimentTable.from_df(experiment_df)
        measurement_table = core.MeasurementTable.from_df(measurement_df)
        mapping_table = core.MappingTable.from_df(mapping_df)
        parameter_table = core.ParameterTable.from_df(parameter_df)

        return Problem(
            model=model,
            condition_table=condition_table,
            experiment_table=experiment_table,
            observable_table=observable_table,
            measurement_table=measurement_table,
            parameter_table=parameter_table,
            mapping_table=mapping_table,
            visualization_df=visualization_df,
            config=config,
        )

    @staticmethod
    def from_combine(filename: Path | str) -> Problem:
        """Read PEtab COMBINE archive (http://co.mbine.org/documents/archive).

        See also :py:func:`petab.v2.create_combine_archive`.

        Arguments:
            filename: Path to the PEtab-COMBINE archive

        Returns:
            A :py:class:`petab.v2.Problem` instance.
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

    @property
    def condition_df(self) -> pd.DataFrame | None:
        """Condition table as DataFrame."""
        # TODO: return empty df?
        return self.condition_table.to_df() if self.condition_table else None

    @condition_df.setter
    def condition_df(self, value: pd.DataFrame):
        self.condition_table = core.ConditionTable.from_df(value)

    @property
    def experiment_df(self) -> pd.DataFrame | None:
        """Experiment table as DataFrame."""
        return self.experiment_table.to_df() if self.experiment_table else None

    @experiment_df.setter
    def experiment_df(self, value: pd.DataFrame):
        self.experiment_table = core.ExperimentTable.from_df(value)

    @property
    def measurement_df(self) -> pd.DataFrame | None:
        """Measurement table as DataFrame."""
        return (
            self.measurement_table.to_df() if self.measurement_table else None
        )

    @measurement_df.setter
    def measurement_df(self, value: pd.DataFrame):
        self.measurement_table = core.MeasurementTable.from_df(value)

    @property
    def parameter_df(self) -> pd.DataFrame | None:
        """Parameter table as DataFrame."""
        return self.parameter_table.to_df() if self.parameter_table else None

    @parameter_df.setter
    def parameter_df(self, value: pd.DataFrame):
        self.parameter_table = core.ParameterTable.from_df(value)

    @property
    def observable_df(self) -> pd.DataFrame | None:
        """Observable table as DataFrame."""
        return self.observable_table.to_df() if self.observable_table else None

    @observable_df.setter
    def observable_df(self, value: pd.DataFrame):
        self.observable_table = core.ObservableTable.from_df(value)

    @property
    def mapping_df(self) -> pd.DataFrame | None:
        """Mapping table as DataFrame."""
        return self.mapping_table.to_df() if self.mapping_table else None

    @mapping_df.setter
    def mapping_df(self, value: pd.DataFrame):
        self.mapping_table = core.MappingTable.from_df(value)

    def get_optimization_parameters(self) -> list[str]:
        """
        Get the list of optimization parameter IDs from parameter table.

        Arguments:
            parameter_df: PEtab parameter DataFrame

        Returns:
            A list of IDs of parameters selected for optimization
            (i.e., those with estimate = True).
        """
        return [p.id for p in self.parameter_table.parameters if p.estimate]

    def get_optimization_parameter_scales(self) -> dict[str, str]:
        """
        Return list of optimization parameter scaling strings.

        See :py:func:`petab.parameters.get_optimization_parameters`.
        """
        # TODO: to be removed in v2?
        return parameters.get_optimization_parameter_scaling(self.parameter_df)

    def get_observable_ids(self) -> list[str]:
        """
        Returns dictionary of observable ids.
        """
        return [o.id for o in self.observable_table.observables]

    def _apply_mask(self, v: list, free: bool = True, fixed: bool = True):
        """Apply mask of only free or only fixed values.

        Parameters
        ----------
        v:
            The full vector the mask is to be applied to.
        free:
            Whether to return free parameters, i.e., parameters to estimate.
        fixed:
            Whether to return fixed parameters, i.e., parameters not to
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
        v = [p.id for p in self.parameter_table.parameters]
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
    ) -> list:
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
        v = [
            p.nominal_value if p.nominal_value is not None else nan
            for p in self.parameter_table.parameters
        ]

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
        v = [
            p.lb if p.lb is not None else nan
            for p in self.parameter_table.parameters
        ]
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
        v = [
            p.ub if p.ub is not None else nan
            for p in self.parameter_table.parameters
        ]
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
        return [
            i
            for i, p in enumerate(self.parameter_table.parameters)
            if p.estimate
        ]

    @property
    def x_fixed_indices(self) -> list[int]:
        """Parameter table non-estimated parameter indices."""
        return [
            i
            for i, p in enumerate(self.parameter_table.parameters)
            if not p.estimate
        ]

    # TODO remove in v2?
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

    def get_priors(self) -> dict[str, Distribution]:
        """Get prior distributions.

        :returns: The prior distributions for the estimated parameters.
        """
        return {
            p.id: p.prior_dist
            for p in self.parameter_table.parameters
            if p.estimate
        }

    def sample_parameter_startpoints(self, n_starts: int = 100, **kwargs):
        """Create 2D array with starting points for optimization"""
        priors = self.get_priors()
        return np.vstack([p.sample(n_starts) for p in priors.values()]).T

    def sample_parameter_startpoints_dict(
        self, n_starts: int = 100
    ) -> list[dict[str, float]]:
        """Create dictionaries with starting points for optimization

        :returns:
            A list of dictionaries with parameter IDs mapping to sampled
            parameter values.
        """
        return [
            dict(zip(self.x_free_ids, parameter_values, strict=True))
            for parameter_values in self.sample_parameter_startpoints(
                n_starts=n_starts
            )
        ]

    # TODO: remove in v2?
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

    # TODO: remove in v2?
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
        return len(self.measurement_table.measurements)

    @property
    def n_priors(self) -> int:
        """Number of priors."""
        return sum(
            p.prior_distribution is not None
            for p in self.parameter_table.parameters
        )

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
        from ..v2.lint import (
            ValidationIssue,
            ValidationIssueSeverity,
            ValidationResultList,
        )

        validation_results = ValidationResultList()
        if self.config and self.config.extensions:
            extensions = ",".join(self.config.extensions.keys())
            validation_results.append(
                ValidationIssue(
                    ValidationIssueSeverity.WARNING,
                    "Validation of PEtab extensions is not yet implemented, "
                    "but the given problem uses the following extensions: "
                    f"{extensions}",
                )
            )

        for task in validation_tasks or self.validation_tasks:
            try:
                cur_result = task.run(self)
            except Exception as e:
                cur_result = ValidationIssue(
                    ValidationIssueSeverity.CRITICAL,
                    f"Validation task {task} failed with exception: {e}\n"
                    f"{traceback.format_exc()}",
                )

            if cur_result:
                validation_results.append(cur_result)

                if cur_result.level == ValidationIssueSeverity.CRITICAL:
                    break

        return validation_results

    def add_condition(
        self, id_: str, name: str = None, **kwargs: Number | str | sp.Expr
    ):
        """Add a simulation condition to the problem.

        Arguments:
            id_: The condition id
            name: The condition name
            kwargs: Entities to be added to the condition table in the form
                `target_id=target_value`.
        """
        if not kwargs:
            raise ValueError("Cannot add condition without any changes")

        changes = [
            core.Change(target_id=target_id, target_value=target_value)
            for target_id, target_value in kwargs.items()
        ]
        self.condition_table.conditions.append(
            core.Condition(id=id_, changes=changes)
        )
        if name is not None:
            self.mapping_table.mappings.append(
                core.Mapping(
                    petab_id=id_,
                    name=name,
                )
            )

    def add_observable(
        self,
        id_: str,
        formula: str,
        noise_formula: str | float | int = None,
        noise_distribution: str = None,
        observable_placeholders: list[str] = None,
        noise_placeholders: list[str] = None,
        name: str = None,
        **kwargs,
    ):
        """Add an observable to the problem.

        Arguments:
            id_: The observable id
            formula: The observable formula
            noise_formula: The noise formula
            noise_distribution: The noise distribution
            observable_placeholders: Placeholders for the observable formula
            noise_placeholders: Placeholders for the noise formula
            name: The observable name
            kwargs: additional columns/values to add to the observable table

        """
        record = {
            OBSERVABLE_ID: id_,
            OBSERVABLE_FORMULA: formula,
        }
        if name is not None:
            record[OBSERVABLE_NAME] = name
        if noise_formula is not None:
            record[NOISE_FORMULA] = noise_formula
        if noise_distribution is not None:
            record[NOISE_DISTRIBUTION] = noise_distribution
        if observable_placeholders is not None:
            record[OBSERVABLE_PLACEHOLDERS] = observable_placeholders
        if noise_placeholders is not None:
            record[NOISE_PLACEHOLDERS] = noise_placeholders
        record.update(kwargs)

        self.observable_table += core.Observable(**record)

    def add_parameter(
        self,
        id_: str,
        estimate: bool | str = True,
        nominal_value: Number | None = None,
        scale: str = None,
        lb: Number = None,
        ub: Number = None,
        prior_dist: str = None,
        prior_pars: str | Sequence = None,
        **kwargs,
    ):
        """Add a parameter to the problem.

        Arguments:
            id_: The parameter id
            estimate: Whether the parameter is estimated
            nominal_value: The nominal value of the parameter
            scale: The parameter scale
            lb: The lower bound of the parameter
            ub: The upper bound of the parameter
            prior_dist: The type of the prior distribution
            prior_pars: The parameters of the prior distribution
            kwargs: additional columns/values to add to the parameter table
        """
        record = {
            PARAMETER_ID: id_,
        }
        if estimate is not None:
            record[ESTIMATE] = estimate
        if nominal_value is not None:
            record[NOMINAL_VALUE] = nominal_value
        if scale is not None:
            record[PARAMETER_SCALE] = scale
        if lb is not None:
            record[LOWER_BOUND] = lb
        if ub is not None:
            record[UPPER_BOUND] = ub
        if prior_dist is not None:
            record[PRIOR_DISTRIBUTION] = prior_dist
        if prior_pars is not None:
            if isinstance(prior_pars, Sequence) and not isinstance(
                prior_pars, str
            ):
                prior_pars = PARAMETER_SEPARATOR.join(map(str, prior_pars))
            record[PRIOR_PARAMETERS] = prior_pars
        record.update(kwargs)

        self.parameter_table += core.Parameter(**record)

    def add_measurement(
        self,
        obs_id: str,
        experiment_id: str,
        time: float,
        measurement: float,
        observable_parameters: Sequence[str | float] | str | float = None,
        noise_parameters: Sequence[str | float] | str | float = None,
    ):
        """Add a measurement to the problem.

        Arguments:
            obs_id: The observable ID
            experiment_id: The experiment ID
            time: The measurement time
            measurement: The measurement value
            observable_parameters: The observable parameters
            noise_parameters: The noise parameters
        """
        if observable_parameters is not None and not isinstance(
            observable_parameters, Sequence
        ):
            observable_parameters = [observable_parameters]
        if noise_parameters is not None and not isinstance(
            noise_parameters, Sequence
        ):
            noise_parameters = [noise_parameters]

        self.measurement_table.measurements.append(
            core.Measurement(
                observable_id=obs_id,
                experiment_id=experiment_id,
                time=time,
                measurement=measurement,
                observable_parameters=observable_parameters,
                noise_parameters=noise_parameters,
            )
        )

    def add_mapping(self, petab_id: str, model_id: str, name: str = None):
        """Add a mapping table entry to the problem.

        Arguments:
            petab_id: The new PEtab-compatible ID mapping to `model_id`
            model_id: The ID of some entity in the model
        """
        self.mapping_table.mappings.append(
            core.Mapping(petab_id=petab_id, model_id=model_id, name=name)
        )

    def add_experiment(self, id_: str, *args):
        """Add an experiment to the problem.

        :param id_: The experiment ID.
        :param args: Timepoints and associated conditions:
            ``time_1, condition_id_1, time_2, condition_id_2, ...``.
        """
        if len(args) % 2 != 0:
            raise ValueError(
                "Arguments must be pairs of timepoints and condition IDs."
            )

        periods = [
            core.ExperimentPeriod(
                time=args[i],
                condition_ids=[cond]
                if isinstance((cond := args[i + 1]), str)
                else cond,
            )
            for i in range(0, len(args), 2)
        ]

        self.experiment_table.experiments.append(
            core.Experiment(id=id_, periods=periods)
        )

    def __iadd__(self, other):
        """Add Observable, Parameter, Measurement, Condition, or Experiment"""
        from .core import (
            Condition,
            Experiment,
            Measurement,
            Observable,
            Parameter,
        )

        if isinstance(other, Observable):
            self.observable_table += other
        elif isinstance(other, Parameter):
            self.parameter_table += other
        elif isinstance(other, Measurement):
            self.measurement_table += other
        elif isinstance(other, Condition):
            self.condition_table += other
        elif isinstance(other, Experiment):
            self.experiment_table += other
        else:
            raise ValueError(
                f"Cannot add object of type {type(other)} to Problem."
            )
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Convert this Problem to a dictionary.

        This function is intended for debugging purposes and should not be
        used for serialization. The output of this function may change
        without notice.

        The output includes all PEtab tables, but not the model itself.

        See `pydantic.BaseModel.model_dump <https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump>`__
        for details.

        :example:

        >>> from pprint import pprint
        >>> p = Problem()
        >>> p += core.Parameter(id="par", lb=0, ub=1)
        >>> pprint(p.model_dump())
        {'conditions': [],
         'config': {'condition_files': [],
                    'experiment_files': [],
                    'extensions': {},
                    'format_version': '2.0.0',
                    'mapping_files': [],
                    'measurement_files': [],
                    'model_files': {},
                    'observable_files': [],
                    'parameter_file': [],
                    'visualization_files': []},
         'experiments': [],
         'mappings': [],
         'measurements': [],
         'observables': [],
         'parameters': [{'estimate': 'true',
                         'id': 'par',
                         'lb': 0.0,
                         'nominal_value': None,
                         'prior_distribution': '',
                         'prior_parameters': '',
                         'ub': 1.0}]}
        """
        res = {
            "config": (self.config or ProblemConfig()).model_dump(
                **kwargs, by_alias=True
            ),
        }
        res |= self.mapping_table.model_dump(**kwargs)
        res |= self.condition_table.model_dump(**kwargs)
        res |= self.experiment_table.model_dump(**kwargs)
        res |= self.observable_table.model_dump(**kwargs)
        res |= self.measurement_table.model_dump(**kwargs)
        res |= self.parameter_table.model_dump(**kwargs)

        return res


class ModelFile(BaseModel):
    """A file in the PEtab problem configuration."""

    location: str | AnyUrl
    language: str


class ExtensionConfig(BaseModel):
    """The configuration of a PEtab extension."""

    version: str
    config: dict


class ProblemConfig(BaseModel):
    """The PEtab problem configuration."""

    #: The path to the PEtab problem configuration.
    filepath: str | AnyUrl | None = Field(
        None,
        description="The path to the PEtab problem configuration.",
        exclude=True,
    )
    #: The base path to resolve relative paths.
    base_path: str | AnyUrl | None = Field(
        None,
        description="The base path to resolve relative paths.",
        exclude=True,
    )
    #: The PEtab format version.
    format_version: str = "2.0.0"
    #: The path to the parameter file, relative to ``base_path``.
    # TODO https://github.com/PEtab-dev/PEtab/pull/641:
    #  rename to parameter_files in yaml for consistency with other files?
    #   always a list?
    parameter_files: list[str | AnyUrl] = Field(
        default=[], alias=PARAMETER_FILE
    )

    # TODO: consider changing str to Path
    model_files: dict[str, ModelFile] | None = {}
    measurement_files: list[str | AnyUrl] = []
    condition_files: list[str | AnyUrl] = []
    experiment_files: list[str | AnyUrl] = []
    observable_files: list[str | AnyUrl] = []
    visualization_files: list[str | AnyUrl] = []
    mapping_files: list[str | AnyUrl] = []

    #: Extensions used by the problem.
    extensions: list[ExtensionConfig] | dict = {}

    # convert parameter_file to list
    @field_validator(
        "parameter_files",
        mode="before",
    )
    def _convert_parameter_file(cls, v):
        """Convert parameter_file to a list."""
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError(
            "parameter_files must be a string or a list of strings."
        )

    def to_yaml(self, filename: str | Path):
        """Write the configuration to a YAML file.

        :param filename: Destination file name. The parent directory will be
            created if necessary.
        """
        from ..v1.yaml import write_yaml

        write_yaml(self.model_dump(by_alias=True), filename)

    @property
    def format_version_tuple(self) -> tuple[int, int, int, str]:
        """The format version as a tuple of major/minor/patch `int`s and a
        suffix."""
        return parse_version(self.format_version)
