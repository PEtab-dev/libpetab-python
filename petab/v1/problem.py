"""PEtab Problem class"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from math import nan
from numbers import Number
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from warnings import warn

import pandas as pd
from pydantic import AnyUrl, BaseModel, Field

from ..versions import get_major_version
from . import (
    conditions,
    core,
    mapping,
    measurements,
    observables,
    parameter_mapping,
    parameters,
    sampling,
    sbml,
    yaml,
)
from .C import *  # noqa: F403
from .models import MODEL_TYPE_SBML
from .models.model import Model, model_factory
from .models.sbml_model import SbmlModel
from .yaml import get_path_prefix

if TYPE_CHECKING:
    import libsbml


__all__ = ["Problem"]


class Problem:
    """
    PEtab parameter estimation problem.

    A PEtab problem as defined by:

    - model
    - condition table
    - measurement table
    - parameter table
    - observables table
    - mapping table

    Optionally, it may contain visualization tables.

    See also :doc:`petab:v1/documentation_data_format`.

    Parameters:
        condition_df: PEtab condition table
        measurement_df: PEtab measurement table
        parameter_df: PEtab parameter table
        observable_df: PEtab observable table
        visualization_df: PEtab visualization table
        mapping_df: PEtab mapping table
        model: The underlying model
        sbml_reader: Stored to keep object alive (deprecated).
        sbml_document: Stored to keep object alive (deprecated).
        sbml_model: PEtab SBML model (deprecated)
        extensions_config: Information on the extensions used
    """

    def __init__(
        self,
        sbml_model: libsbml.Model = None,
        sbml_reader: libsbml.SBMLReader = None,
        sbml_document: libsbml.SBMLDocument = None,
        model: Model = None,
        model_id: str = None,
        condition_df: pd.DataFrame = None,
        measurement_df: pd.DataFrame = None,
        parameter_df: pd.DataFrame = None,
        visualization_df: pd.DataFrame = None,
        observable_df: pd.DataFrame = None,
        mapping_df: pd.DataFrame = None,
        extensions_config: dict = None,
        config: ProblemConfig = None,
    ):
        self.condition_df: pd.DataFrame | None = condition_df
        self.measurement_df: pd.DataFrame | None = measurement_df
        self.parameter_df: pd.DataFrame | None = parameter_df
        self.visualization_df: pd.DataFrame | None = visualization_df
        self.observable_df: pd.DataFrame | None = observable_df
        self.mapping_df: pd.DataFrame | None = mapping_df

        if any(
            (sbml_model, sbml_document, sbml_reader),
        ):
            warn(
                "Passing `sbml_model`, `sbml_document`, or `sbml_reader` "
                "to petab.Problem is deprecated and will be removed in a "
                "future version. Use `model=petab.models.sbml_model."
                "SbmlModel(...)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if model:
                raise ValueError(
                    "Must only provide one of (`sbml_model`, "
                    "`sbml_document`, `sbml_reader`) or `model`."
                )

            model = SbmlModel(
                sbml_model=sbml_model,
                sbml_reader=sbml_reader,
                sbml_document=sbml_document,
                model_id=model_id,
            )

        self.model: Model | None = model
        self.extensions_config = extensions_config or {}
        self.config = config

    def __getattr__(self, name):
        # For backward-compatibility, allow access to SBML model related
        #  attributes now stored in self.model
        if name in {"sbml_model", "sbml_reader", "sbml_document"}:
            return getattr(self.model, name) if self.model else None
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        # For backward-compatibility, allow access to SBML model related
        #  attributes now stored in self.model
        if name in {"sbml_model", "sbml_reader", "sbml_document"}:
            if self.model:
                setattr(self.model, name, value)
            else:
                self.model = SbmlModel(**{name: value})
        else:
            super().__setattr__(name, value)

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
    def from_files(
        sbml_file: str | Path = None,
        condition_file: str | Path | Iterable[str | Path] = None,
        measurement_file: str | Path | Iterable[str | Path] = None,
        parameter_file: str | Path | Iterable[str | Path] = None,
        visualization_files: str | Path | Iterable[str | Path] = None,
        observable_files: str | Path | Iterable[str | Path] = None,
        model_id: str = None,
        extensions_config: dict = None,
    ) -> Problem:
        """
        Factory method to load model and tables from files.

        Arguments:
            sbml_file: PEtab SBML model
            condition_file: PEtab condition table
            measurement_file: PEtab measurement table
            parameter_file: PEtab parameter table
            visualization_files: PEtab visualization tables
            observable_files: PEtab observables tables
            model_id: PEtab ID of the model
            extensions_config: Information on the extensions used
        """
        warn(
            "petab.Problem.from_files is deprecated and will be removed in a "
            "future version. Use `petab.Problem.from_yaml instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        model = (
            model_factory(sbml_file, MODEL_TYPE_SBML, model_id=model_id)
            if sbml_file
            else None
        )

        condition_df = (
            core.concat_tables(condition_file, conditions.get_condition_df)
            if condition_file
            else None
        )

        # If there are multiple tables, we will merge them
        measurement_df = (
            core.concat_tables(
                measurement_file, measurements.get_measurement_df
            )
            if measurement_file
            else None
        )

        parameter_df = (
            parameters.get_parameter_df(parameter_file)
            if parameter_file
            else None
        )

        # If there are multiple tables, we will merge them
        visualization_df = (
            core.concat_tables(visualization_files, core.get_visualization_df)
            if visualization_files
            else None
        )

        # If there are multiple tables, we will merge them
        observable_df = (
            core.concat_tables(observable_files, observables.get_observable_df)
            if observable_files
            else None
        )

        return Problem(
            model=model,
            condition_df=condition_df,
            measurement_df=measurement_df,
            parameter_df=parameter_df,
            observable_df=observable_df,
            visualization_df=visualization_df,
            extensions_config=extensions_config,
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
        # path to the yaml file
        filepath = None

        if isinstance(yaml_config, Path):
            yaml_config = str(yaml_config)

        if isinstance(yaml_config, str):
            filepath = yaml_config
            if base_path is None:
                base_path = get_path_prefix(yaml_config)
            yaml_config = yaml.load_yaml(yaml_config)

        def get_path(filename):
            if base_path is None:
                return filename
            return f"{base_path}/{filename}"

        if yaml.is_composite_problem(yaml_config):
            raise ValueError(
                "petab.Problem.from_yaml() can only be used for "
                "yaml files comprising a single model. "
                "Consider using "
                "petab.CompositeProblem.from_yaml() instead."
            )

        major_version = get_major_version(yaml_config)
        if major_version not in {1, 2}:
            raise ValueError(
                "Provided PEtab files are of unsupported version "
                f"{yaml_config[FORMAT_VERSION]}."
            )
        if major_version == 2:
            warn(
                "Using petab.v1.Problem with PEtab2.0 is deprecated. "
                "Use petab.v2.Problem instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        config = ProblemConfig(
            **yaml_config, base_path=base_path, filepath=filepath
        )
        problem0 = config.problems[0]
        # currently required for handling PEtab v2 in here
        problem0_ = yaml_config["problems"][0]

        if isinstance(config.parameter_file, list):
            parameter_df = parameters.get_parameter_df(
                [get_path(f) for f in config.parameter_file]
            )
        else:
            parameter_df = (
                parameters.get_parameter_df(get_path(config.parameter_file))
                if config.parameter_file
                else None
            )
        if major_version == 1:
            if len(problem0.sbml_files) > 1:
                # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
                raise NotImplementedError(
                    "Support for multiple models is not yet implemented."
                )

            model = (
                model_factory(
                    get_path(problem0.sbml_files[0]),
                    MODEL_TYPE_SBML,
                    model_id=None,
                )
                if problem0.sbml_files
                else None
            )
        else:
            if len(problem0_[MODEL_FILES]) > 1:
                # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
                raise NotImplementedError(
                    "Support for multiple models is not yet implemented."
                )
            if not problem0_[MODEL_FILES]:
                model = None
            else:
                model_id, model_info = next(
                    iter(problem0_[MODEL_FILES].items())
                )
                model = model_factory(
                    get_path(model_info[MODEL_LOCATION]),
                    model_info[MODEL_LANGUAGE],
                    model_id=model_id,
                )

        measurement_files = [get_path(f) for f in problem0.measurement_files]
        # If there are multiple tables, we will merge them
        measurement_df = (
            core.concat_tables(
                measurement_files, measurements.get_measurement_df
            )
            if measurement_files
            else None
        )

        condition_files = [get_path(f) for f in problem0.condition_files]
        # If there are multiple tables, we will merge them
        condition_df = (
            core.concat_tables(condition_files, conditions.get_condition_df)
            if condition_files
            else None
        )

        visualization_files = [
            get_path(f) for f in problem0.visualization_files
        ]
        # If there are multiple tables, we will merge them
        visualization_df = (
            core.concat_tables(visualization_files, core.get_visualization_df)
            if visualization_files
            else None
        )

        observable_files = [get_path(f) for f in problem0.observable_files]
        # If there are multiple tables, we will merge them
        observable_df = (
            core.concat_tables(observable_files, observables.get_observable_df)
            if observable_files
            else None
        )

        mapping_files = [get_path(f) for f in problem0_.get(MAPPING_FILES, [])]
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
            config=config,
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

    def to_files_generic(
        self,
        prefix_path: str | Path,
    ) -> str:
        """Save a PEtab problem to generic file names.

        The PEtab problem YAML file is always created. PEtab data files are
        only created if the PEtab problem contains corresponding data (e.g. a
        PEtab visualization TSV file is only created if the PEtab problem has
        one).

        Arguments:
            prefix_path:
                Specify a prefix to all paths, to avoid specifying the
                prefix for all paths individually. NB: the prefix is added to
                paths before ``relative_paths`` is handled downstream in
                :func:`petab.yaml.create_problem_yaml`.

        Returns:
            The path to the PEtab problem YAML file.
        """
        prefix_path = Path(prefix_path)

        # Generate generic filenames for data tables in the PEtab problem that
        # contain data.
        filenames = {}
        for table_name in [
            "condition",
            "measurement",
            "parameter",
            "observable",
            "visualization",
            "mapping",
        ]:
            if getattr(self, f"{table_name}_df") is not None:
                filenames[f"{table_name}_file"] = f"{table_name}s.tsv"

        if self.model:
            if not isinstance(self.model, SbmlModel):
                raise NotImplementedError(
                    "Saving non-SBML models is currently not supported."
                )
            filenames["model_file"] = "model.xml"

        filenames["yaml_file"] = "problem.yaml"

        self.to_files(**filenames, prefix_path=prefix_path)

        if prefix_path is None:
            return filenames["yaml_file"]
        return str(PurePosixPath(prefix_path, filenames["yaml_file"]))

    def to_files(
        self,
        sbml_file: None | str | Path = None,
        condition_file: None | str | Path = None,
        measurement_file: None | str | Path = None,
        parameter_file: None | str | Path = None,
        visualization_file: None | str | Path = None,
        observable_file: None | str | Path = None,
        yaml_file: None | str | Path = None,
        prefix_path: None | str | Path = None,
        relative_paths: bool = True,
        model_file: None | str | Path = None,
        mapping_file: None | str | Path = None,
    ) -> None:
        """
        Write PEtab tables to files for this problem

        Writes PEtab files for those entities for which a destination was
        passed.

        NOTE: If this instance was created from multiple measurement or
        visualization tables, they will be merged and written to a single file.

        Arguments:
            sbml_file: SBML model destination (deprecated)
            model_file: Model destination
            condition_file: Condition table destination
            measurement_file: Measurement table destination
            parameter_file: Parameter table destination
            visualization_file: Visualization table destination
            observable_file: Observables table destination
            mapping_file: Mapping table destination
            yaml_file: YAML file destination
            prefix_path:
                Specify a prefix to all paths, to avoid specifying the
                prefix for all paths individually. NB: the prefix is added to
                paths before ``relative_paths`` is handled.
            relative_paths:
                whether all paths in the YAML file should be
                relative to the location of the YAML file. If ``False``, then
                paths are left unchanged.

        Raises:
            ValueError:
                If a destination was provided for a non-existing entity.
        """
        if sbml_file:
            warn(
                "The `sbml_file` argument is deprecated and will be "
                "removed in a future version. Use `model_file` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            if model_file:
                raise ValueError(
                    "Must provide either `sbml_file` or "
                    "`model_file` argument, but not both."
                )

            model_file = sbml_file

        if prefix_path is not None:
            prefix_path = Path(prefix_path)

            def add_prefix(path0: None | str | Path) -> str:
                return path0 if path0 is None else str(prefix_path / path0)

            model_file = add_prefix(model_file)
            condition_file = add_prefix(condition_file)
            measurement_file = add_prefix(measurement_file)
            parameter_file = add_prefix(parameter_file)
            observable_file = add_prefix(observable_file)
            visualization_file = add_prefix(visualization_file)
            mapping_file = add_prefix(mapping_file)
            yaml_file = add_prefix(yaml_file)

        if model_file:
            self.model.to_file(model_file)

        def error(name: str) -> ValueError:
            return ValueError(f"Unable to save non-existent {name} table")

        if condition_file:
            if self.condition_df is not None:
                conditions.write_condition_df(
                    self.condition_df, condition_file
                )
            else:
                raise error("condition")

        if measurement_file:
            if self.measurement_df is not None:
                measurements.write_measurement_df(
                    self.measurement_df, measurement_file
                )
            else:
                raise error("measurement")

        if parameter_file:
            if self.parameter_df is not None:
                parameters.write_parameter_df(
                    self.parameter_df, parameter_file
                )
            else:
                raise error("parameter")

        if observable_file:
            if self.observable_df is not None:
                observables.write_observable_df(
                    self.observable_df, observable_file
                )
            else:
                raise error("observable")

        if visualization_file:
            if self.visualization_df is not None:
                core.write_visualization_df(
                    self.visualization_df, visualization_file
                )
            else:
                raise error("visualization")

        if mapping_file:
            if self.mapping_df is not None:
                mapping.write_mapping_df(self.mapping_df, mapping_file)
            else:
                raise error("mapping")

        if yaml_file:
            yaml.create_problem_yaml(
                sbml_files=model_file,
                condition_files=condition_file,
                measurement_files=measurement_file,
                parameter_file=parameter_file,
                observable_files=observable_file,
                yaml_file=yaml_file,
                visualization_files=visualization_file,
                relative_paths=relative_paths,
                mapping_files=mapping_file,
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

    def get_model_parameters(self) -> list[str] | dict[str, float]:
        """See :py:func:`petab.sbml.get_model_parameters`"""
        warn(
            "petab.Problem.get_model_parameters is deprecated and will be "
            "removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        return sbml.get_model_parameters(self.sbml_model)

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

    def get_simulation_conditions_from_measurement_df(self):
        """See petab.get_simulation_conditions"""
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

    def create_parameter_df(self, **kwargs):
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

    def add_condition(self, id_: str, name: str = None, **kwargs):
        """Add a simulation condition to the problem.

        Arguments:
            id_: The condition id
            name: The condition name
            kwargs: Parameter, value pairs to add to the condition table.
        """
        record = {CONDITION_ID: [id_], **kwargs}
        if name is not None:
            record[CONDITION_NAME] = name
        tmp_df = pd.DataFrame(record).set_index([CONDITION_ID])
        self.condition_df = (
            pd.concat([self.condition_df, tmp_df])
            if self.condition_df is not None
            else tmp_df
        )

    def add_observable(
        self,
        id_: str,
        formula: str | float | int,
        noise_formula: str | float | int = None,
        noise_distribution: str = None,
        transform: str = None,
        name: str = None,
        **kwargs,
    ):
        """Add an observable to the problem.

        Arguments:
            id_: The observable id
            formula: The observable formula
            noise_formula: The noise formula
            noise_distribution: The noise distribution
            transform: The observable transformation
            name: The observable name
            kwargs: additional columns/values to add to the observable table

        """
        record = {
            OBSERVABLE_ID: [id_],
            OBSERVABLE_FORMULA: [formula],
        }
        if name is not None:
            record[OBSERVABLE_NAME] = [name]
        if noise_formula is not None:
            record[NOISE_FORMULA] = [noise_formula]
        if noise_distribution is not None:
            record[NOISE_DISTRIBUTION] = [noise_distribution]
        if transform is not None:
            record[OBSERVABLE_TRANSFORMATION] = [transform]
        record.update(kwargs)

        tmp_df = pd.DataFrame(record).set_index([OBSERVABLE_ID])
        self.observable_df = (
            pd.concat([self.observable_df, tmp_df])
            if self.observable_df is not None
            else tmp_df
        )

    def add_parameter(
        self,
        id_: str,
        estimate: bool | str | int = True,
        nominal_value: Number | None = None,
        scale: str = None,
        lb: Number = None,
        ub: Number = None,
        init_prior_type: str = None,
        init_prior_pars: str | Sequence = None,
        obj_prior_type: str = None,
        obj_prior_pars: str | Sequence = None,
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
            init_prior_type: The type of the initialization prior distribution
            init_prior_pars: The parameters of the initialization prior
                distribution
            obj_prior_type: The type of the objective prior distribution
            obj_prior_pars: The parameters of the objective prior distribution
            kwargs: additional columns/values to add to the parameter table
        """
        record = {
            PARAMETER_ID: [id_],
        }
        if estimate is not None:
            record[ESTIMATE] = [int(estimate)]
        if nominal_value is not None:
            record[NOMINAL_VALUE] = [nominal_value]
        if scale is not None:
            record[PARAMETER_SCALE] = [scale]
        if lb is not None:
            record[LOWER_BOUND] = [lb]
        if ub is not None:
            record[UPPER_BOUND] = [ub]
        if init_prior_type is not None:
            record[INITIALIZATION_PRIOR_TYPE] = [init_prior_type]
        if init_prior_pars is not None:
            if not isinstance(init_prior_pars, str):
                init_prior_pars = PARAMETER_SEPARATOR.join(
                    map(str, init_prior_pars)
                )
            record[INITIALIZATION_PRIOR_PARAMETERS] = [init_prior_pars]
        if obj_prior_type is not None:
            record[OBJECTIVE_PRIOR_TYPE] = [obj_prior_type]
        if obj_prior_pars is not None:
            if not isinstance(obj_prior_pars, str):
                obj_prior_pars = PARAMETER_SEPARATOR.join(
                    map(str, obj_prior_pars)
                )
            record[OBJECTIVE_PRIOR_PARAMETERS] = [obj_prior_pars]
        record.update(kwargs)

        tmp_df = pd.DataFrame(record).set_index([PARAMETER_ID])
        self.parameter_df = (
            pd.concat([self.parameter_df, tmp_df])
            if self.parameter_df is not None
            else tmp_df
        )

    def add_measurement(
        self,
        obs_id: str,
        sim_cond_id: str,
        time: float,
        measurement: float,
        observable_parameters: Sequence[str | float] = None,
        noise_parameters: Sequence[str | float] = None,
        preeq_cond_id: str = None,
    ):
        """Add a measurement to the problem.

        Arguments:
            obs_id: The observable ID
            sim_cond_id: The simulation condition ID
            time: The measurement time
            measurement: The measurement value
            observable_parameters: The observable parameters
            noise_parameters: The noise parameters
            preeq_cond_id: The pre-equilibration condition ID
        """
        record = {
            OBSERVABLE_ID: [obs_id],
            SIMULATION_CONDITION_ID: [sim_cond_id],
            TIME: [time],
            MEASUREMENT: [measurement],
        }
        if observable_parameters is not None:
            record[OBSERVABLE_PARAMETERS] = [
                PARAMETER_SEPARATOR.join(map(str, observable_parameters))
            ]
        if noise_parameters is not None:
            record[NOISE_PARAMETERS] = [
                PARAMETER_SEPARATOR.join(map(str, noise_parameters))
            ]
        if preeq_cond_id is not None:
            record[PREEQUILIBRATION_CONDITION_ID] = [preeq_cond_id]

        tmp_df = pd.DataFrame(record)
        self.measurement_df = (
            pd.concat([self.measurement_df, tmp_df])
            if self.measurement_df is not None
            else tmp_df
        )


class SubProblem(BaseModel):
    """A `problems` object in the PEtab problem configuration."""

    sbml_files: list[str | AnyUrl] = []
    measurement_files: list[str | AnyUrl] = []
    condition_files: list[str | AnyUrl] = []
    observable_files: list[str | AnyUrl] = []
    visualization_files: list[str | AnyUrl] = []


class ProblemConfig(BaseModel):
    """The PEtab problem configuration."""

    filepath: str | AnyUrl | None = Field(
        None,
        description="The path to the PEtab problem configuration.",
        exclude=True,
    )
    base_path: str | AnyUrl | None = Field(
        None,
        description="The base path to resolve relative paths.",
        exclude=True,
    )
    format_version: str | int = 1
    parameter_file: str | AnyUrl | None = None
    problems: list[SubProblem] = []

    def to_yaml(self, filename: str | Path):
        """Write the configuration to a YAML file.

        :param filename: Destination file name. The parent directory will be
            created if necessary.
        """
        from .yaml import write_yaml

        write_yaml(self.model_dump(), filename)
