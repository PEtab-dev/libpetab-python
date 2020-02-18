# create the parameter file in the PEtab format

import pandas as pd
import os
import libsedml
import libsbml
import numpy as np
from decimal import Decimal


def parameterPETAB(new_sbml_save_path, sedml_file_name, measdatafile_save_path):

    # create new folder
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/parameters'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/parameters')

    # save path
    correct_petab_name = 'parameters_' + sedml_file_name + '.tsv'
    parameter_save_path = './sedml2petab/' + sedml_file_name + '/parameters/' + correct_petab_name

    # create new data frame
    ParFile = pd.DataFrame(columns=['parameterId', 'parameterName', 'parameterScale', 'lowerBound',
                                    'upperBound', 'nominalValue', 'estimate', 'priorType',
                                    'priorParameters', 'HierarchicalOptimization (optional)'], data=[])

    # open sbml model and measurement_data file to collect data about all parameters
    sbml_model = libsbml.readSBML(new_sbml_save_path)
    MesDataFile = pd.read_csv(measdatafile_save_path, sep='\t')
    par_list = []
    par_value = []
    lin_log_list = []
    for iPar in range(0, sbml_model.getModel().getNumParameters()):
        if sbml_model.getModel().getParameter(iPar).getMetaId() != '':
            par_list.append(sbml_model.getModel().getParameter(iPar).getId())
            if sbml_model.getModel().getParameter(iPar).getValue() != 0:
                #par_value.append(np.log10(sbml_model.getModel().getParameter(iPar).getValue()))
                par_value.append(sbml_model.getModel().getParameter(iPar).getValue())
                lin_log_list.append('log10')
            elif sbml_model.getModel().getParameter(iPar).getValue() == 0:
                par_value.append(sbml_model.getModel().getParameter(iPar).getValue())
                lin_log_list.append('lin')

    # get sigma names from MesDataFile only once
    sigma_name = []
    for iElement in range(0, len(MesDataFile['observableId'])):
        if iElement == 0:
            sigma_name.append('sigma_' + MesDataFile['observableId'][0])
        else:
            if MesDataFile['observableId'][iElement] != MesDataFile['observableId'][iElement - 1]:
                sigma_name.append('sigma_' + MesDataFile['observableId'][iElement])

    # get additional 'observableParameters' from the MesDataFile
    additional_observables = []
    for iElement in range(0, len(MesDataFile['observableParameters'])):
        if iElement == 0:
            additional_observables.append(MesDataFile['observableParameters'][0])
        else:
            if MesDataFile['observableParameters'][iElement] != MesDataFile['observableParameters'][iElement - 1]:
                additional_observables.append(MesDataFile['observableParameters'][iElement])

    # get corresponding values from sbml file
    corresponding_values = []
    new_additional_observables = []
    for iObs in range(0, len(additional_observables)):
        if len(additional_observables[iObs]) == 1:
            new_additional_observables.append(additional_observables[iObs])
            for iPar in range(0, sbml_model.getModel().getNumParameters()):
                if additional_observables[iObs] == sbml_model.getModel().getParameter(iPar).getId():
                    if sbml_model.getModel().getParameter(iPar).getValue() != 0:
                        corresponding_values.append(np.log10(sbml_model.getModel().getParameter(iPar).getValue()))
                    else:
                        corresponding_values.append(0)
                    break
        else:
            subList = additional_observables[iObs].split(';')
            for iSubObs in range(0, len(subList)):
                new_additional_observables.append(subList[iSubObs])
                for iPar in range(0, sbml_model.getModel().getNumParameters()):
                    if subList[iSubObs] == sbml_model.getModel().getParameter(iPar).getId():
                        if sbml_model.getModel().getParameter(iPar).getValue() != 0:
                            #corresponding_values.append(np.log10(sbml_model.getModel().getParameter(iPar).getValue()))
                            corresponding_values.append(sbml_model.getModel().getParameter(iPar).getValue())
                        else:
                            corresponding_values.append(0)
                        break

    # assign 'lin' or 'log10' according to the value
    more_lin_log = []
    for iValue in range(0, len(corresponding_values)):
        if corresponding_values[iValue] == 0:
            more_lin_log.append('lin')
        else:
            more_lin_log.append('log10')

    # use new data to fill in the new data frame
    # unused columns can simply remain empty
    ParFile['parameterId'] = pd.concat([pd.Series(par_list), pd.Series(sigma_name), pd.Series(new_additional_observables)], ignore_index=True)
    ParFile['parameterName'] = ParFile['parameterId']
    ParFile['nominalValue'] = pd.concat([pd.Series(par_value), pd.Series([4] * len(sigma_name)), pd.Series(corresponding_values)], ignore_index=True)

    # possibly it has to be user defined
    ParFile['parameterScale'] = pd.concat([pd.Series(lin_log_list), pd.Series(['log10'] * len(sigma_name)), pd.Series(more_lin_log)], ignore_index=True)
    ParFile['lowerBound'] = pd.Series(["{:.0E}".format(Decimal(f'{10**(-10)}'))] * len(ParFile['parameterId']))
    ParFile['upperBound'] = pd.Series(["{:.0E}".format(Decimal(f'{10**10}'))] * len(ParFile['parameterId']))
    ParFile['estimate'] = pd.Series(['1'] * len(ParFile['parameterId']))

    # save data frame as .tsv
    ParFile.to_csv(parameter_save_path, sep='\t', index=False)

    return parameter_save_path