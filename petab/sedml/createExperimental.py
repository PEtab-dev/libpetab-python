# create the experimental condition file in the PEtab format
# the amount of conditions depends on the amount of sbml files

import pandas as pd
import os
import libsedml


def experimentalPETAB(sedml_save_path, sedml_file_name):

    # create new folder
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/experimental_condition'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/experimental_condition')

    # save path
    correct_petab_name = 'experimentalCondition_' + sedml_file_name + '.tsv'
    experimental_save_path = './sedml2petab/' + sedml_file_name + '/experimental_condition/' + correct_petab_name

    # create new data frame
    ExpConFile = pd.DataFrame(columns=['conditionId', 'conditionName'], data=[])

    # extend data frame regarding the amount of sbml models
    sedml_model = libsedml.readSedML(sedml_save_path)
    if sedml_model.getNumModels() == 1:
        sbml_Id = sedml_model.getModel(0).getId()
        ExpConFile = ExpConFile.append({}, ignore_index=True)

        # assign the id to the data frame
        ExpConFile['conditionId'][0] = 'condition1'
        ExpConFile['conditionName'][0] = sbml_Id

    # save data frame as .tsv
    ExpConFile.to_csv(experimental_save_path, sep='\t', index=False)

    return experimental_save_path