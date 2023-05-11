"""PEtab Problem class"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Union, TYPE_CHECKING
from urllib.parse import unquote, urlparse, urlunparse
from warnings import warn

import pandas as pd

from . import (conditions, core, format_version, measurements, observables,
               parameter_mapping, parameters, sampling, sbml, yaml, mapping)
from .C import *  # noqa: F403
from .models import MODEL_TYPE_SBML
from .models.model import Model, model_factory
from .models.sbml_model import SbmlModel

if TYPE_CHECKING:
    import libsbml


__all__ = ['Problem']


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

    Attributes:
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
            extensions_config: Dict = None,
    ):
        self.condition_df: Optional[pd.DataFrame] = condition_df
        self.measurement_df: Optional[pd.DataFrame] = measurement_df
        self.parameter_df: Optional[pd.DataFrame] = parameter_df
        self.visualization_df: Optional[pd.DataFrame] = visualization_df
        self.observable_df: Optional[pd.DataFrame] = observable_df
        self.mapping_df: Optional[pd.DataFrame] = mapping_df

        if any((sbml_model, sbml_document, sbml_reader),):
            warn("Passing `sbml_model`, `sbml_document`, or `sbml_reader` "
                 "to petab.Problem is deprecated and will be removed in a "
                 "future version. Use `model=petab.models.sbml_model."
                 "SbmlModel(...)` instead.", DeprecationWarning, stacklevel=2)
            if model:
                raise ValueError("Must only provide one of (`sbml_model`, "
                                 "`sbml_document`, `sbml_reader`) or `model`.")

            model = SbmlModel(
                sbml_model=sbml_model,
                sbml_reader=sbml_reader,
                sbml_document=sbml_document,
                model_id=model_id
            )

        self.model: Optional[Model] = model
        self.extensions_config = extensions_config or {}

    def __getattr__(self, name):
        # For backward-compatibility, allow access to SBML model related
        #  attributes now stored in self.model
        if name in {'sbml_model', 'sbml_reader', 'sbml_document'}:
            return getattr(self.model, name) if self.model else None
        raise AttributeError(f"'{self.__class__.__name__}' object has no "
                             f"attribute '{name}'")

    def __setattr__(self, name, value):
        # For backward-compatibility, allow access to SBML model related
        #  attributes now stored in self.model
        if name in {'sbml_model', 'sbml_reader', 'sbml_document'}:
            if self.model:
                setattr(self.model, name, value)
            else:
                self.model = SbmlModel(**{name: value})
        else:
            super().__setattr__(name, value)

    def __str__(self):
        model = f"with model ({self.model})" if self.model else "without model"
        conditions = f"{self.condition_df.shape[0]} conditions" \
            if self.condition_df is not None else "without conditions table"

        observables = f"{self.observable_df.shape[0]} observables" \
            if self.observable_df is not None else "without observables table"

        measurements = f"{self.measurement_df.shape[0]} measurements" \
            if self.measurement_df is not None \
            else "without measurements table"

        if self.parameter_df is not None:
            num_estimated_parameters = sum(self.parameter_df[ESTIMATE] == 1) \
                if ESTIMATE in self.parameter_df \
                else self.parameter_df.shape[0]
            parameters = f"{num_estimated_parameters} estimated parameters"
        else:
            parameters = "without parameter_df table"

        return (
            f"PEtab Problem {model}, {conditions}, {observables}, "
            f"{measurements}, {parameters}"
        )

    @staticmethod
    def from_files(
            sbml_file: Union[str, Path] = None,
            condition_file:
            Union[str, Path, Iterable[Union[str, Path]]] = None,
            measurement_file: Union[str, Path,
                                    Iterable[Union[str, Path]]] = None,
            parameter_file: Union[str, Path,
                                  Iterable[Union[str, Path]]] = None,
            visualization_files: Union[str, Path,
                                       Iterable[Union[str, Path]]] = None,
            observable_files: Union[str, Path,
                                    Iterable[Union[str, Path]]] = None,
            model_id: str = None,
            extensions_config: Dict = None,
    ) -> 'Problem':
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
        warn("petab.Problem.from_files is deprecated and will be removed in a "
             "future version. Use `petab.Problem.from_yaml instead.",
             DeprecationWarning, stacklevel=2)

        model = model_factory(sbml_file, MODEL_TYPE_SBML, model_id=model_id) \
            if sbml_file else None

        condition_df = core.concat_tables(
            condition_file, conditions.get_condition_df) \
            if condition_file else None

        # If there are multiple tables, we will merge them
        measurement_df = core.concat_tables(
            measurement_file, measurements.get_measurement_df) \
            if measurement_file else None

        parameter_df = parameters.get_parameter_df(parameter_file) \
            if parameter_file else None

        # If there are multiple tables, we will merge them
        visualization_df = core.concat_tables(
            visualization_files, core.get_visualization_df) \
            if visualization_files else None

        # If there are multiple tables, we will merge them
        observable_df = core.concat_tables(
            observable_files, observables.get_observable_df) \
            if observable_files else None

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
    def from_yaml(yaml_config: Union[Dict, Path, str]) -> 'Problem':
        """
        Factory method to load model and tables as specified by YAML file.

        Arguments:
            yaml_config: PEtab configuration as dictionary or YAML file name
        """
        if isinstance(yaml_config, Path):
            yaml_config = str(yaml_config)

        get_path = lambda filename: filename  # noqa: E731
        if isinstance(yaml_config, str):
            yaml_path = yaml_config
            yaml_config = yaml.load_yaml(yaml_config)

            # yaml_config may be path or URL
            path_url = urlparse(yaml_path)
            if not path_url.scheme or \
                    (path_url.scheme != 'file' and not path_url.netloc):
                # a regular file path string
                path_prefix = Path(yaml_path).parent
                get_path = lambda filename: \
                    path_prefix / filename  # noqa: E731
            else:
                # a URL
                # extract parent path from
                url_path = unquote(urlparse(yaml_path).path)
                parent_path = str(PurePosixPath(url_path).parent)
                path_prefix = urlunparse(
                    (path_url.scheme, path_url.netloc, parent_path,
                     path_url.params, path_url.query, path_url.fragment)
                )
                # need "/" on windows, not "\"
                get_path = lambda filename: \
                    f"{path_prefix}/{filename}"  # noqa: E731

        if yaml.is_composite_problem(yaml_config):
            raise ValueError('petab.Problem.from_yaml() can only be used for '
                             'yaml files comprising a single model. '
                             'Consider using '
                             'petab.CompositeProblem.from_yaml() instead.')

        if yaml_config[FORMAT_VERSION] not in {"1", 1, "1.0.0", "2.0.0"}:
            raise ValueError("Provided PEtab files are of unsupported version "
                             f"{yaml_config[FORMAT_VERSION]}. Expected "
                             f"{format_version.__format_version__}.")
        if yaml_config[FORMAT_VERSION] == "2.0.0":
            warn("Support for PEtab2.0 is experimental!")

        problem0 = yaml_config['problems'][0]

        if isinstance(yaml_config[PARAMETER_FILE], list):
            parameter_df = parameters.get_parameter_df([
                get_path(f)
                for f in yaml_config[PARAMETER_FILE]
            ])
        else:
            parameter_df = parameters.get_parameter_df(
                get_path(yaml_config[PARAMETER_FILE])) \
                if yaml_config[PARAMETER_FILE] else None

        if yaml_config[FORMAT_VERSION] in [1, "1", "1.0.0"]:
            if len(problem0[SBML_FILES]) > 1:
                # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
                raise NotImplementedError(
                    'Support for multiple models is not yet implemented.')

            model = model_factory(get_path(problem0[SBML_FILES][0]),
                                  MODEL_TYPE_SBML, model_id=None) \
                if problem0[SBML_FILES] else None
        else:
            if len(problem0[MODEL_FILES]) > 1:
                # TODO https://github.com/PEtab-dev/libpetab-python/issues/6
                raise NotImplementedError(
                    'Support for multiple models is not yet implemented.')
            if not problem0[MODEL_FILES]:
                model = None
            else:
                model_id, model_info = \
                    next(iter(problem0[MODEL_FILES].items()))
                model = model_factory(get_path(model_info[MODEL_LOCATION]),
                                      model_info[MODEL_LANGUAGE],
                                      model_id=model_id)

        measurement_files = [
            get_path(f) for f in problem0.get(MEASUREMENT_FILES, [])]
        # If there are multiple tables, we will merge them
        measurement_df = core.concat_tables(
            measurement_files, measurements.get_measurement_df) \
            if measurement_files else None

        condition_files = [
            get_path(f) for f in problem0.get(CONDITION_FILES, [])]
        # If there are multiple tables, we will merge them
        condition_df = core.concat_tables(
            condition_files, conditions.get_condition_df) \
            if condition_files else None

        visualization_files = [
            get_path(f) for f in problem0.get(VISUALIZATION_FILES, [])]
        # If there are multiple tables, we will merge them
        visualization_df = core.concat_tables(
            visualization_files, core.get_visualization_df) \
            if visualization_files else None

        observable_files = [
            get_path(f) for f in problem0.get(OBSERVABLE_FILES, [])]
        # If there are multiple tables, we will merge them
        observable_df = core.concat_tables(
            observable_files, observables.get_observable_df) \
            if observable_files else None

        mapping_files = [
            get_path(f) for f in problem0.get(MAPPING_FILES, [])]
        # If there are multiple tables, we will merge them
        mapping_df = core.concat_tables(
            mapping_files, mapping.get_mapping_df) \
            if mapping_files else None

        return Problem(
            condition_df=condition_df,
            measurement_df=measurement_df,
            parameter_df=parameter_df,
            observable_df=observable_df,
            model=model,
            visualization_df=visualization_df,
            mapping_df=mapping_df,
            extensions_config=yaml_config.get(EXTENSIONS, {})
        )

    @staticmethod
    def from_combine(filename: Union[Path, str]) -> 'Problem':
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
                "(python-libcombine) must be installed.") from e

        archive = libcombine.CombineArchive()
        if archive.initializeFromArchive(str(filename)) is None:
            print(f"Invalid Combine Archive: {filename}")
            return None

        with tempfile.TemporaryDirectory() as tmpdirname:
            archive.extractTo(tmpdirname)
            problem = Problem.from_yaml(
                os.path.join(tmpdirname,
                             archive.getMasterFile().getLocation()))
        archive.cleanUp()

        return problem

    def to_files_generic(
        self,
        prefix_path: Union[str, Path],
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
            'condition',
            'measurement',
            'parameter',
            'observable',
            'visualization',
            'mapping',
        ]:
            if getattr(self, f'{table_name}_df') is not None:
                filenames[f'{table_name}_file'] = f'{table_name}s.tsv'

        if self.model:
            if not isinstance(self.model, SbmlModel):
                raise NotImplementedError("Saving non-SBML models is "
                                          "currently not supported.")
            filenames['model_file'] = 'model.xml'

        filenames['yaml_file'] = 'problem.yaml'

        self.to_files(**filenames, prefix_path=prefix_path)

        if prefix_path is None:
            return filenames['yaml_file']
        return str(prefix_path / filenames['yaml_file'])

    def to_files(
            self,
            sbml_file: Union[None, str, Path] = None,
            condition_file: Union[None, str, Path] = None,
            measurement_file: Union[None, str, Path] = None,
            parameter_file: Union[None, str, Path] = None,
            visualization_file: Union[None, str, Path] = None,
            observable_file: Union[None, str, Path] = None,
            yaml_file: Union[None, str, Path] = None,
            prefix_path: Union[None, str, Path] = None,
            relative_paths: bool = True,
            model_file: Union[None, str, Path] = None,
            mapping_file: Union[None, str, Path] = None,
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
            warn("The `sbml_file` argument is deprecated and will be "
                 "removed in a future version. Use `model_file` instead.",
                 DeprecationWarning, stacklevel=2)

            if model_file:
                raise ValueError("Must provide either `sbml_file` or "
                                 "`model_file` argument, but not both.")

            model_file = sbml_file

        if prefix_path is not None:
            prefix_path = Path(prefix_path)

            def add_prefix(path0: Union[None, str, Path]) -> str:
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
                conditions.write_condition_df(self.condition_df,
                                              condition_file)
            else:
                raise error("condition")

        if measurement_file:
            if self.measurement_df is not None:
                measurements.write_measurement_df(self.measurement_df,
                                                  measurement_file)
            else:
                raise error("measurement")

        if parameter_file:
            if self.parameter_df is not None:
                parameters.write_parameter_df(self.parameter_df,
                                              parameter_file)
            else:
                raise error("parameter")

        if observable_file:
            if self.observable_df is not None:
                observables.write_observable_df(self.observable_df,
                                                observable_file)
            else:
                raise error("observable")

        if visualization_file:
            if self.visualization_df is not None:
                core.write_visualization_df(self.visualization_df,
                                            visualization_file)
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

    def get_optimization_parameters(self):
        """
        Return list of optimization parameter IDs.

        See :py:func:`petab.parameters.get_optimization_parameters`.
        """
        return parameters.get_optimization_parameters(self.parameter_df)

    def get_optimization_parameter_scales(self):
        """
        Return list of optimization parameter scaling strings.

        See :py:func:`petab.parameters.get_optimization_parameters`.
        """
        return parameters.get_optimization_parameter_scaling(self.parameter_df)

    def get_model_parameters(self):
        """See :py:func:`petab.sbml.get_model_parameters`"""
        warn("petab.Problem.get_model_parameters is deprecated and will be "
             "removed in a future version.",
             DeprecationWarning, stacklevel=2)

        return sbml.get_model_parameters(self.sbml_model)

    def get_observable_ids(self):
        """
        Returns dictionary of observable ids.
        """
        return list(self.observable_df.index)

    def _apply_mask(self, v: List, free: bool = True, fixed: bool = True):
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
    def x_ids(self) -> List[str]:
        """Parameter table parameter IDs"""
        return self.get_x_ids()

    @property
    def x_free_ids(self) -> List[str]:
        """Parameter table parameter IDs, for free parameters."""
        return self.get_x_ids(fixed=False)

    @property
    def x_fixed_ids(self) -> List[str]:
        """Parameter table parameter IDs, for fixed parameters."""
        return self.get_x_ids(free=False)

    def get_x_nominal(self, free: bool = True, fixed: bool = True,
                      scaled: bool = False):
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
        v = list(self.parameter_df[NOMINAL_VALUE])
        if scaled:
            v = list(parameters.map_scale(
                v, self.parameter_df[PARAMETER_SCALE]))
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def x_nominal(self) -> List:
        """Parameter table nominal values"""
        return self.get_x_nominal()

    @property
    def x_nominal_free(self) -> List:
        """Parameter table nominal values, for free parameters."""
        return self.get_x_nominal(fixed=False)

    @property
    def x_nominal_fixed(self) -> List:
        """Parameter table nominal values, for fixed parameters."""
        return self.get_x_nominal(free=False)

    @property
    def x_nominal_scaled(self) -> List:
        """Parameter table nominal values with applied parameter scaling"""
        return self.get_x_nominal(scaled=True)

    @property
    def x_nominal_free_scaled(self) -> List:
        """Parameter table nominal values with applied parameter scaling,
        for free parameters."""
        return self.get_x_nominal(fixed=False, scaled=True)

    @property
    def x_nominal_fixed_scaled(self) -> List:
        """Parameter table nominal values with applied parameter scaling,
        for fixed parameters."""
        return self.get_x_nominal(free=False, scaled=True)

    def get_lb(self, free: bool = True, fixed: bool = True,
               scaled: bool = False):
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
            v = list(parameters.map_scale(
                v, self.parameter_df[PARAMETER_SCALE]))
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def lb(self) -> List:
        """Parameter table lower bounds."""
        return self.get_lb()

    @property
    def lb_scaled(self) -> List:
        """Parameter table lower bounds with applied parameter scaling"""
        return self.get_lb(scaled=True)

    def get_ub(self, free: bool = True, fixed: bool = True,
               scaled: bool = False):
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
            v = list(parameters.map_scale(
                v, self.parameter_df[PARAMETER_SCALE]))
        return self._apply_mask(v, free=free, fixed=fixed)

    @property
    def ub(self) -> List:
        """Parameter table upper bounds"""
        return self.get_ub()

    @property
    def ub_scaled(self) -> List:
        """Parameter table upper bounds with applied parameter scaling"""
        return self.get_ub(scaled=True)

    @property
    def x_free_indices(self) -> List[int]:
        """Parameter table estimated parameter indices."""
        estimated = list(self.parameter_df[ESTIMATE])
        return [j for j, val in enumerate(estimated) if val != 0]

    @property
    def x_fixed_indices(self) -> List[int]:
        """Parameter table non-estimated parameter indices."""
        estimated = list(self.parameter_df[ESTIMATE])
        return [j for j, val in enumerate(estimated) if val == 0]

    def get_simulation_conditions_from_measurement_df(self):
        """See petab.get_simulation_conditions"""
        return measurements.get_simulation_conditions(self.measurement_df)

    def get_optimization_to_simulation_parameter_mapping(
            self, **kwargs
    ):
        """
        See
        :py:func:`petab.parameter_mapping.get_optimization_to_simulation_parameter_mapping`,
        to which all keyword arguments are forwarded.
        """
        return parameter_mapping \
            .get_optimization_to_simulation_parameter_mapping(
                condition_df=self.condition_df,
                measurement_df=self.measurement_df,
                parameter_df=self.parameter_df,
                observable_df=self.observable_df,
                model=self.model,
                **kwargs
            )

    def create_parameter_df(self, *args, **kwargs):
        """Create a new PEtab parameter table

        See :py:func:`create_parameter_df`.
        """
        return parameters.create_parameter_df(
            model=self.model,
            condition_df=self.condition_df,
            observable_df=self.observable_df,
            measurement_df=self.measurement_df,
            mapping_df=self.mapping_df,
            *args, **kwargs)

    def sample_parameter_startpoints(self, n_starts: int = 100):
        """Create 2D array with starting points for optimization

        See :py:func:`petab.sample_parameter_startpoints`.
        """
        return sampling.sample_parameter_startpoints(
            self.parameter_df, n_starts=n_starts)

    def sample_parameter_startpoints_dict(
            self,
            n_starts: int = 100
    ) -> List[Dict[str, float]]:
        """Create dictionaries with starting points for optimization

        See also :py:func:`petab.sample_parameter_startpoints`.

        Returns:
            A list of dictionaries with parameter IDs mapping to samples
            parameter values.
        """
        return [
            dict(zip(self.x_free_ids, parameter_values))
            for parameter_values in self.sample_parameter_startpoints(
                n_starts=n_starts
            )
        ]

    def unscale_parameters(
        self,
        x_dict: Dict[str, float],
    ) -> Dict[str, float]:
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
        x_dict: Dict[str, float],
    ) -> Dict[str, float]:
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
