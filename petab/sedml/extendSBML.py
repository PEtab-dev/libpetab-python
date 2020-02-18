# extend the sbml_file_with_observables by 'noise_', 'sigma_', 'observable_' as [parameters] and [assignment_rules]

import libsbml
import os
import pandas as pd
import importlib
from .changeObservableId import *


def editSBML(new_sbml_save_path, sedml_file_name, parameters_save_path):

    # reload libsbml to work around an error
    importlib.reload(libsbml)

    # create new folder
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/sbml_models_with_observables_and_assignment_rules'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/sbml_models_with_observables_and_assignment_rules')

    # open SBMl model
    sbml_file = libsbml.readSBML(new_sbml_save_path)
    sbml_model = sbml_file.getModel()

    # read parameters file for the sigma_parameters
    sigma_parameters = []
    parameters_file = pd.read_csv(parameters_save_path, sep='\t')
    for iPar in range(0, len(parameters_file['parameterId'])):
        if 'sigma_' in parameters_file['parameterId'][iPar]:
            sigma_parameters.append(parameters_file['parameterId'][iPar])
            #sigma_parameters = sigma_parameters.reset_index(drop=True)
            #break

    # transform all sigma_parameter into a 'sigma_', 'noiseParameter1_' and 'observable_' parameter
    noise_sigma_observable = []
    for iElement in range(0, len(sigma_parameters)):
        noise_sigma_observable.append(sigma_parameters[iElement])
        noise_sigma_observable.append('noiseParameter1_' + sigma_parameters[iElement].split('sigma_')[1])
        noise_sigma_observable.append('observable_' + sigma_parameters[iElement].split('sigma_')[1])

    # change the observable names to the petab format default names
    sbml_file = changeObsId(sbml_file, sigma_parameters)

    # create new parameters
    for iPar in range(0, len(noise_sigma_observable)):
        p = sbml_model.createParameter()
        p.setId(noise_sigma_observable[iPar])
        p.setName(noise_sigma_observable[iPar])
        p.setConstant(False)
        p.setValue(1)                                                                                                   # why 1?

    # create new assignment rule
    for iAssignmentRule in range(0, len(noise_sigma_observable)):
        if 'sigma_' in noise_sigma_observable[iAssignmentRule]:
            rule = sbml_model.createAssignmentRule()
            rule.setId(noise_sigma_observable[iAssignmentRule])
            rule.setName(noise_sigma_observable[iAssignmentRule])
            rule.setVariable(noise_sigma_observable[iAssignmentRule])
            rule.setFormula(noise_sigma_observable[iAssignmentRule + 1])
        elif 'noiseParameter1_' in noise_sigma_observable[iAssignmentRule]:
            continue
        '''
        elif 'observable_' in noise_sigma_observable[iAssignmentRule]:
            rule = sbml_model.createAssignmentRule()
            rule.setId(noise_sigma_observable[iAssignmentRule])
            rule.setName(noise_sigma_observable[iAssignmentRule])
            rule.setVariable(noise_sigma_observable[iAssignmentRule])

            # get the correct species name for the value
            _,species = noise_sigma_observable[iAssignmentRule].split('observable_')
            for iSpecies in range(0, sbml_model.getNumSpecies()):
                if species == sbml_model.getSpecies(iSpecies).getId():
                    rule.setFormula(species)
                    break
                elif iSpecies == sbml_model.getNumSpecies() - 1:
                    rule.setFormula(sbml_model.getSpecies(0).getId())
        '''

    # save new sbml model
    newest_sbml_save_path = './sedml2petab/' + sedml_file_name + '/sbml_models_with_observables_and_assignment_rules/model_' + sedml_file_name + '.xml'
    libsbml.writeSBMLToFile(sbml_file, newest_sbml_save_path)


    return newest_sbml_save_path