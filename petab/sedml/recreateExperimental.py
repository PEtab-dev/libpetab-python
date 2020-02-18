# recreate the original experimental data file

import pandas as pd
import os
import re
import sys
import shutil


def recreateExpDataFile(petab_folder_path):                                                                             #exp_save_path, meas_save_path, par_save_path, sbml_save_path):

    # create new folder 'petab2sedml'
    if not os.path.exists('./petab2sedml'):
        os.makedirs('./petab2sedml')

    # get the name of the model from the sbml model
    all_files = sorted(os.listdir(petab_folder_path))
    for Files in all_files:
        if 'model_' in Files:
            _,sbml_file_name,_ = re.split('model_|\.',Files)

    # new base path
    base_path = './petab2sedml/' + sbml_file_name

    # copy all files into new folder
    if not os.path.isdir(base_path + '/old_' + sbml_file_name):
        shutil.copytree(petab_folder_path, base_path + '/old_' + sbml_file_name)

    # create new folder for the recreated exp_con_data file
    if not os.path.exists(base_path + '/experimental_data'):
        os.makedirs(base_path + '/experimental_data')

    # all new important paths
    new_sbml_save_path = base_path + '/old_' + sbml_file_name + '/model_' + sbml_file_name + '.xml'
    new_exp_save_path = base_path + '/old_' + sbml_file_name + '/experimentalCondition_' + sbml_file_name + '.tsv'
    new_meas_save_path = base_path + '/old_' + sbml_file_name + '/measurementData_' + sbml_file_name + '.tsv'
    new_par_save_path = base_path + '/old_' + sbml_file_name + '/parameters_' + sbml_file_name + '.tsv'

    # load all three .tsv files
    experimental_condition_file = pd.read_csv(new_exp_save_path, sep='\t')
    measurement_file = pd.read_csv(new_meas_save_path, sep='\t')
    parameters_file = pd.read_csv(new_par_save_path, sep='\t')

    # get column names
    id_list = ['time']
    id_list.append(measurement_file['observableId'][0])
    for iElement in range(1, len(measurement_file['observableId'])):
        if measurement_file['observableId'][iElement] != measurement_file['observableId'][iElement - 1]:
            id_list.append(measurement_file['observableId'][iElement])
            if len(id_list) == 3:
                first_change = iElement

    # build new data frame
    exp_con_data = pd.DataFrame(columns=id_list, data=[])

    # sort and extract all important information
    unique_times = sorted(list(set(measurement_file['time'])))
    unique_observables = sorted(list(set(measurement_file['observableId'])))
    if not unique_observables == sorted(id_list[1:]):
        print('Column names do not match with the observables!')
        sys.exit()
    exp_con_data['time'] = pd.Series(unique_times)
    for iColumn in range(0, len(unique_observables)):
        intersect_list = []
        for iRow in range(0, len(unique_times)):
            intersect = measurement_file.loc[(measurement_file['time'] == unique_times[iRow]) & (measurement_file['observableId'] == unique_observables[iColumn]), 'measurement']
            #times = measurement_file.loc[measurement_file['time'] == unique_times[iRow], 'measurement']
            #observables = measurement_file.loc[measurement_file['observableId'] == unique_observables[iColumn], 'measurement']
            #intersect = list(set(list(times)).intersection(set((list(observables)))))
            if len(intersect) == 1:
                intersect_list.append(list(intersect)[0])
            elif len(intersect) == 0:
                print('No match between measurement and timepoint!')
                sys.exit()
            elif len(intersect) > 1:
                print('Two measurements for the same time point --- Multiple experimental condition files necessary!')
        exp_con_data[unique_observables[iColumn]] = pd.Series(intersect_list)


    # save new data frame
    exp_con_save_path = [base_path + '/experimental_data/' + sbml_file_name + '_model.tsv']
    for iExpDataFile in range(0, len(exp_con_save_path)):
        exp_con_data.to_csv(exp_con_save_path[iExpDataFile], sep='\t', index=False)

    return sbml_file_name, base_path, exp_con_save_path, new_exp_save_path, new_meas_save_path, new_par_save_path, new_sbml_save_path