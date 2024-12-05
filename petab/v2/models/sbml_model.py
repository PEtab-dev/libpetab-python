"""Functions for handling SBML models"""
import libsbml

from ...v1.models.sbml_model import *  # noqa: F401, F403
from ..experiments import Experiment
from ..parameter_mapping import get_parameter_mapping_for_condition


def get_period_model(
    petab_problem: "petab.Problem",
    experiment_id: str,
    period_index: int,
) -> tuple[libsbml.SBMLDocument, libsbml.Model]:
    """Create an SBML model for the given period.

    Creates a copy of the model and updates parameters according to the PEtab
    files. Estimated parameters are set to their ``nominalValue``.
    Observables defined in the observables table are not added to the model.

    :param petab_problem: PEtab problem
    :param experiment_id: The experiment that contains the period.
    :param period_index: The index of the period in the experiment period
        sequence. N.B.: this is the index in the denested period sequence.
    :return: The generated SBML document, and SBML model
    """
    assert isinstance(petab_problem.model, SbmlModel)

    experiment = Experiment.from_df(
        experiment_df=petab_problem.experiment_df,
        experiment_id=experiment_id,
    )

    period = experiment.periods[period_index]

    condition_id = period.condition_id
    period_measurement_df = period.get_measurements(
        measurement_df=petab_problem.measurement_df
    )

    (
        parameter_map,
        scale_map,
    ) = get_parameter_mapping_for_condition(
        condition_id=condition_id,
        is_initial_period=period_index == 0,
        cur_measurement_df=period_measurement_df,
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
                    parameter_id, petab.NOMINAL_VALUE
                ]

        if not isinstance(mapped_value, str):
            return mapped_value

        # estimated parameter, look up in nominal parameters
        return petab_problem.parameter_df.loc[
            mapped_value, petab.NOMINAL_VALUE
        ]

    def remove_rules(target_id: str):
        if sbml_model.removeRuleByVariable(target_id):
            warn(
                "An SBML rule was removed to set the component "
                f"{target_id} to a constant value.",
                stacklevel=2,
            )
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
            petab_problem.condition_df.loc[condition_id, component_id]
        )
        if not isinstance(new_value, Number):
            # parameter reference in condition table
            new_value = get_param_value(new_value)

        if sbml_species.isSetInitialAmount() or (
            sbml_species.getHasOnlySubstanceUnits()
            and not sbml_species.isSetInitialConcentration()
        ):
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
            petab_problem.condition_df.loc[condition_id, component_id]
        )
        if not isinstance(new_value, Number):
            # parameter reference in condition table
            new_value = get_param_value(new_value)

        sbml_compartment.setSize(new_value)

    return sbml_doc, sbml_model
