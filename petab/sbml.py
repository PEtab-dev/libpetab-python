"""Functions for interacting with SBML models"""

import contextlib
import logging
from numbers import Number
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from warnings import warn

import libsbml
from pandas.io.common import get_handle, is_file_like, is_url

import petab

logger = logging.getLogger(__name__)
__all__ = [
    'get_model_for_condition',
    'get_model_parameters',
    'get_sbml_model',
    'globalize_parameters',
    'is_sbml_consistent',
    'load_sbml_from_file',
    'load_sbml_from_string',
    'log_sbml_errors',
    'write_sbml'
]


def is_sbml_consistent(
        sbml_document: libsbml.SBMLDocument,
        check_units: bool = False,
) -> bool:
    """Check for SBML validity / consistency

    Arguments:
        sbml_document: SBML document to check
        check_units: Also check for unit-related issues

    Returns:
        ``False`` if problems were detected, otherwise ``True``
    """

    if not check_units:
        sbml_document.setConsistencyChecks(
            libsbml.LIBSBML_CAT_UNITS_CONSISTENCY, False)

    has_problems = sbml_document.checkConsistency()
    if has_problems:
        log_sbml_errors(sbml_document)
        logger.warning(
            'WARNING: Generated invalid SBML model. Check messages above.')

    return not has_problems


def log_sbml_errors(
        sbml_document: libsbml.SBMLDocument,
        minimum_severity=libsbml.LIBSBML_SEV_WARNING,
) -> None:
    """Log libsbml errors

    Arguments:
        sbml_document: SBML document to check
        minimum_severity: Minimum severity level to report (see libsbml
            documentation)
    """
    severity_to_log_level = {
        libsbml.LIBSBML_SEV_INFO: logging.INFO,
        libsbml.LIBSBML_SEV_WARNING: logging.WARNING,
    }
    for error_idx in range(sbml_document.getNumErrors()):
        error = sbml_document.getError(error_idx)
        if (severity := error.getSeverity()) >= minimum_severity:
            category = error.getCategoryAsString()
            severity_str = error.getSeverityAsString()
            message = error.getMessage()
            logger.log(severity_to_log_level.get(severity, logging.ERROR),
                       f'libSBML {severity_str} ({category}): {message}')


def globalize_parameters(
        sbml_model: libsbml.Model,
        prepend_reaction_id: bool = False,
) -> None:
    """Turn all local parameters into global parameters with the same
    properties

    Local parameters are currently ignored by other PEtab functions. Use this
    function to convert them to global parameters. There may exist local
    parameters with identical IDs within different kinetic laws. This is not
    checked here. If in doubt that local parameter IDs are unique, enable
    ``prepend_reaction_id`` to create global parameters named
    ``${reaction_id}_${local_parameter_id}``.

    Arguments:
        sbml_model:
            The SBML model to operate on
        prepend_reaction_id:
            Prepend reaction id of local parameter when
            creating global parameters
    """
    warn("This function will be removed in future releases.",
         DeprecationWarning)

    for reaction in sbml_model.getListOfReactions():
        law = reaction.getKineticLaw()
        # copy first so we can delete in the following loop
        local_parameters = list(local_parameter for local_parameter
                                in law.getListOfParameters())
        for lp in local_parameters:
            if prepend_reaction_id:
                parameter_id = f'{reaction.getId()}_{lp.getId()}'
            else:
                parameter_id = lp.getId()

            # Create global
            p = sbml_model.createParameter()
            p.setId(parameter_id)
            p.setName(lp.getName())
            p.setConstant(lp.getConstant())
            p.setValue(lp.getValue())
            p.setUnits(lp.getUnits())

            # removeParameter, not removeLocalParameter!
            law.removeParameter(lp.getId())


def get_model_parameters(
        sbml_model: libsbml.Model, with_values=False
) -> Union[List[str], Dict[str, float]]:
    """Return SBML model parameters which are not Rule targets

    Arguments:
        sbml_model: SBML model
        with_values:
            If ``False``, returns list of SBML model parameter IDs which
            are not Rule targets. If ``True``, returns a dictionary with those
            parameter IDs as keys and parameter values from the SBML model as
            values.
    """
    if not with_values:
        return [p.getId() for p in sbml_model.getListOfParameters()
                if sbml_model.getRuleByVariable(p.getId()) is None]

    return {p.getId(): p.getValue()
            for p in sbml_model.getListOfParameters()
            if sbml_model.getRuleByVariable(p.getId()) is None}


def write_sbml(
        sbml_doc: libsbml.SBMLDocument,
        filename: Union[Path, str]
) -> None:
    """Write PEtab visualization table

    Arguments:
        sbml_doc: SBML document containing the SBML model
        filename: Destination file name
    """
    sbml_writer = libsbml.SBMLWriter()
    ret = sbml_writer.writeSBMLToFile(sbml_doc, str(filename))
    if not ret:
        raise RuntimeError(f"libSBML reported error {ret} when trying to "
                           f"create SBML file {filename}.")


def get_sbml_model(
        filepath_or_buffer
) -> Tuple[libsbml.SBMLReader, libsbml.SBMLDocument, libsbml.Model]:
    """Get an SBML model from file or URL or file handle

    :param filepath_or_buffer:
        File or URL or file handle to read the model from
    :return: The SBML document, model and reader
    """
    if is_file_like(filepath_or_buffer) or is_url(filepath_or_buffer):
        with get_handle(filepath_or_buffer, mode='r') as io_handle:
            data = load_sbml_from_string(''.join(io_handle.handle))
        # URL or already opened file, we will load the model from a string
        return data

    return load_sbml_from_file(filepath_or_buffer)


def load_sbml_from_string(
        sbml_string: str
) -> Tuple[libsbml.SBMLReader, libsbml.SBMLDocument, libsbml.Model]:
    """Load SBML model from string

    :param sbml_string: Model as XML string
    :return: The SBML document, model and reader
    """

    sbml_reader = libsbml.SBMLReader()
    sbml_document = \
        sbml_reader.readSBMLFromString(sbml_string)
    sbml_model = sbml_document.getModel()

    return sbml_reader, sbml_document, sbml_model


def load_sbml_from_file(
        sbml_file: str
) -> Tuple[libsbml.SBMLReader, libsbml.SBMLDocument, libsbml.Model]:
    """Load SBML model from file

    :param sbml_file: Filename of the SBML file
    :return: The SBML reader, document, model
    """
    sbml_reader = libsbml.SBMLReader()
    sbml_document = sbml_reader.readSBML(sbml_file)
    sbml_model = sbml_document.getModel()

    return sbml_reader, sbml_document, sbml_model


def get_model_for_condition(
        petab_problem: "petab.Problem",
        sim_condition_id: str = None,
        preeq_condition_id: Optional[str] = None,
) -> Tuple[libsbml.SBMLDocument, libsbml.Model]:
    """Create an SBML model for the given condition.

    Creates a copy of the model and updates parameters according to the PEtab
    files. Estimated parameters are set to their ``nominalValue``.
    Observables defined in the observables table are not added to the model.

    :param petab_problem: PEtab problem
    :param sim_condition_id: Simulation ``conditionId`` for which to generate a
        model
    :param preeq_condition_id: Preequilibration ``conditionId`` of the settings
        for which to generate a model. This is only used to determine the
        relevant output parameter overrides. Preequilibration is not encoded
        in the resulting model.
    :return: The generated SBML document, and SBML model
    """
    from .models.sbml_model import SbmlModel
    assert isinstance(petab_problem.model, SbmlModel)

    condition_dict = {petab.SIMULATION_CONDITION_ID: sim_condition_id}
    if preeq_condition_id:
        condition_dict[petab.PREEQUILIBRATION_CONDITION_ID] = \
            preeq_condition_id
    cur_measurement_df = petab.measurements.get_rows_for_condition(
        measurement_df=petab_problem.measurement_df,
        condition=condition_dict,
    )
    parameter_map, scale_map = \
        petab.parameter_mapping.get_parameter_mapping_for_condition(
            condition_id=sim_condition_id,
            is_preeq=False,
            cur_measurement_df=cur_measurement_df,
            model=petab_problem.model,
            condition_df=petab_problem.condition_df,
            parameter_df=petab_problem.parameter_df,
            warn_unmapped=True,
            scaled_parameters=False,
            fill_fixed_parameters=True,
            # will only become problematic once the observable and noise terms
            #  are added to the model
            allow_timepoint_specific_numeric_noise_parameters=True,
        )
    # create a copy of the model
    sbml_doc = petab_problem.model.sbml_model.getSBMLDocument().clone()
    sbml_model = sbml_doc.getModel()

    # fill in parameters
    def get_param_value(parameter_id: str):
        """Parameter value from mapping or nominal value"""
        mapped_value = parameter_map.get(parameter_id)
        if mapped_value is None:
            # Handle parametric initial concentrations
            with contextlib.suppress(KeyError):
                return petab_problem.parameter_df.loc[
                    parameter_id, petab.NOMINAL_VALUE]

        if not isinstance(mapped_value, str):
            return mapped_value

        # estimated parameter, look up in nominal parameters
        return petab_problem.parameter_df.loc[mapped_value,
                                              petab.NOMINAL_VALUE]

    def remove_rules(target_id: str):
        if sbml_model.removeRuleByVariable(target_id):
            warn("An SBML rule was removed to set the component "
                 f"{target_id} to a constant value.")
        sbml_model.removeInitialAssignment(target_id)

    for parameter in sbml_model.getListOfParameters():
        new_value = get_param_value(parameter.getId())
        if new_value:
            parameter.setValue(new_value)
            # remove rules that would override that value
            remove_rules(parameter.getId())

    # set concentrations for any overridden species
    for component_id in petab_problem.condition_df:
        sbml_species = sbml_model.getSpecies(component_id)
        if not sbml_species:
            continue

        # remove any rules overriding that species' initials
        remove_rules(component_id)

        # set initial concentration/amount
        new_value = petab.to_float_if_float(
            petab_problem.condition_df.loc[sim_condition_id, component_id])
        if not isinstance(new_value, Number):
            # parameter reference in condition table
            new_value = get_param_value(new_value)

        if sbml_species.isSetInitialAmount() \
            or (sbml_species.getHasOnlySubstanceUnits()
                and not sbml_species.isSetInitialConcentration()):
            sbml_species.setInitialAmount(new_value)
        else:
            sbml_species.setInitialConcentration(new_value)

    # set compartment size for any compartments in the condition table
    for component_id in petab_problem.condition_df:
        sbml_compartment = sbml_model.getCompartment(component_id)
        if not sbml_compartment:
            continue

        # remove any rules overriding that compartment's size
        remove_rules(component_id)

        # set initial concentration/amount
        new_value = petab.to_float_if_float(
            petab_problem.condition_df.loc[sim_condition_id, component_id])
        if not isinstance(new_value, Number):
            # parameter reference in condition table
            new_value = get_param_value(new_value)

        sbml_compartment.setSize(new_value)

    return sbml_doc, sbml_model
