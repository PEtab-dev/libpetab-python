# create the measurement data file in the PEtab format

import pandas as pd
import os


def measurementPETAB(exp_rearranged_save_path, sedml_file_name):

    # create new folder
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/measurement_data'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/measurement_data')

    # save path
    correct_petab_name = 'measurementData_' + sedml_file_name + '.tsv'
    measurement_save_path = './sedml2petab/' + sedml_file_name + '/measurement_data/' + correct_petab_name

    # create new data frame
    MeasDataFile = pd.DataFrame(columns=['observableId', 'preequilibrationConditionId', 'simulationConditionId',
                                         'measurement', 'time', 'observableParameters', 'noiseParameters',
                                         'observableTransformation', 'noiseDistribution'], data=[])

    # open rearranged experimental condition file to fill in the new data frame
    # unused columns can simply remain empty
    exp_rearranged = pd.read_csv(exp_rearranged_save_path, sep='\t')
    MeasDataFile['observableId'] = exp_rearranged['observableId']
    MeasDataFile['measurement'] = exp_rearranged['measurement']
    MeasDataFile['time'] = exp_rearranged['time']
    MeasDataFile['observableParameters'] = exp_rearranged['observableParameters']
    MeasDataFile['simulationConditionId'] = pd.Series(['condition1'] * len(MeasDataFile['observableId']))
    MeasDataFile['noiseDistribution'] = pd.Series(['normal'] * len(MeasDataFile['observableId']))
    for iElement in range(0, len(MeasDataFile['observableId'])):
        noise = 'sigma_' + str(MeasDataFile['observableId'][iElement])
        MeasDataFile.at[iElement, 'noiseParameters'] = noise                                                               # returns lots of warnings --- rewrite?

    # possible it has to be user-defined
    MeasDataFile['observableTransformation'] = pd.Series(['log10'] * len(MeasDataFile['observableId']))

    # save data frame as .tsv
    MeasDataFile.to_csv(measurement_save_path, sep='\t', index=False)

    return measurement_save_path