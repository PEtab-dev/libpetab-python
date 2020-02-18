# script to rearrange experimnetal data .csv file

import os
import pandas as pd
import libsedml
import sys


def rearrange2PEtab(sedml_path, sedml_file_name):

    # create new folder for all new dataframes
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/experimental_data_rearranged'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/experimental_data_rearranged')

    if os.path.exists('./sedml2petab/' + sedml_file_name + '/experimental_data'):
        list_directory_expdata = sorted(os.listdir('./sedml2petab/' + sedml_file_name + '/experimental_data'))

        for iData in list_directory_expdata:                                                                            # each exp_data_frame

            # build/reset new data frame
            df = pd.DataFrame(columns=['observableId', 'preequilibrationConditionId', 'simulationConditionId',
                                       'measurement', 'time', 'observableParameters', 'noiseParameters',
                                       'observableTransformation', 'noiseDistribution'], data=[])

            # read .xls file
            xls_file_path = './sedml2petab/' + sedml_file_name + '/experimental_data/' + iData
            expdata_name, rest = iData.split('.',1)

            # count number of sheets in excel file
            xls_file = pd.ExcelFile(xls_file_path)
            if len(xls_file.sheet_names) > 1:
                expdata_file = pd.read_excel(xls_file, 'Data')

                # time column
                try:
                    time_data = expdata_file['time']
                except:
                    all_columns = expdata_file.columns
                    smth_with_time = all_columns[0]
                    time_data = expdata_file[smth_with_time]

                if str(time_data[0]).isdigit() == False:
                    del time_data[0]
                    time_data.reset_index(inplace=True, drop=True)

                # get column names for data frame
                counter = 1
                columns = list(expdata_file.columns)
                for iCol in columns:
                    try:
                        name = iCol.split('.')
                        if len(name) == 2:
                            if name[0] == 'time':
                                counter = counter + 1
                    except:
                        counter = 1

            else:
                expdata_file = pd.read_excel(xls_file_path)

                # time column
                try:
                    time_data = expdata_file['time']
                except:
                    all_columns = expdata_file.columns
                    smth_with_time = all_columns[0]
                    time_data = expdata_file[smth_with_time]

                if str(time_data[0]).isdigit() == False:
                    del time_data[0]
                    time_data.reset_index(inplace=True, drop=True)

                # get column names for data frame
                counter = 1
                columns = list(expdata_file.columns)

            for iDataFrame in range(0, len(columns) - counter):                                                               # each data frame to merge together

                # build new data frame
                df_new = pd.DataFrame(columns=['observableId', 'preequilibrationConditionId', 'simulationConditionId',
                                               'measurement', 'time', 'observableParameters', 'noiseParameters',
                                               'observableTransformation', 'noiseDistribution'], data=[])

                ############# get input
                ###### species
                new_species = []
                for iNumber in range(0, len(time_data)):
                    new_species.append(columns[iDataFrame + counter])
                new_species = pd.Series(new_species)

                ###### measurement + reindex from e.g. [1:14] to [0:13]
                new_measurement = expdata_file[columns[iDataFrame + counter]]
                if str(new_measurement[0]).isdigit() == False:
                    del new_measurement[0]
                    new_measurement.reset_index(inplace=True, drop=True)

                ###### observable parameters
                new_observables = []

                sedml_file = libsedml.readSedML(sedml_path)

                # get number of tasks and observables
                num_task = sedml_file.getNumTasks()
                num_obs = sedml_file.getNumDataGenerators()

                #for iTask in range(0, num_task):                                                                        # each task with task_referance [no tasks needed]
                    #task_id = sedml_file.getTask(iTask).getId()

                # create list with all parameter-ids to check for uniqueness
                almost_all_par_id = []

                # check if parameter in data generators even exist + if not, write pd.Series(NaN)
                all_observables = []
                for iObservable in range(0, num_obs):
                    num_par_all = sedml_file.getDataGenerator(iObservable).getNumParameters()
                    all_observables.append(num_par_all)

                if sum(all_observables) == 0:
                    for iNan in range(0, len(new_species)):
                        new_observables.append('NaN')
                else:

                    for iObservable in range(0, num_obs):                                                               # each observable from data generator
                        # get important formula
                        obs_Formula = libsedml.formulaToString(sedml_file.getDataGenerator(iObservable).getMath())
                        obs_Id = sedml_file.getDataGenerator(iObservable).getId()
                        # SBML_model_Id,Observable_Id = obs_Id.split('_',1)
                        new_obs_Id = 'observable_' + obs_Id

                        # get list of parameters
                        list_par_id = []
                        list_par_value = []
                        num_par = sedml_file.getDataGenerator(iObservable).getNumParameters()

                        # get observable name
                        data_generator_name = sedml_file.getDataGenerator(iObservable).getName()

                        if num_par == 0:                                                                                # parameter from each observable
                            print(sedml_file_name + '_' + iData + '_' + obs_Id + ' has no parameters as observables!')
                            #new_observables.append(pd.Series('NaN'))
                            #if data_generator_name == new_species[0]:                                                   ########## error: some sedml observables don't have a name
                                #for iNumber in range(0, len(time_data)):
                                    #new_observables.append(pd.Series('NaN'))
                                #new_observables = pd.Series(new_observables)

                        else:
                            for iCount in range(0, num_par):                                                            # add parameters + values
                                list_par_id.append(sedml_file.getDataGenerator(iObservable).getParameter(iCount).getId())
                                list_par_value.append(sedml_file.getDataGenerator(iObservable).getParameter(iCount).getValue())
                                almost_all_par_id.append(sedml_file.getDataGenerator(iObservable).getParameter(iCount).getId())

                                # check for uniqueness of parameter-ids
                                for iNum in range(0, len(almost_all_par_id)):                                           # unique?
                                    all_par_id = almost_all_par_id[iNum]
                                    almost_all_par_id.remove(almost_all_par_id[len(almost_all_par_id) - 1])
                                    last_element = list(all_par_id[len(all_par_id) - 1])
                                    intersection = [i for i in last_element if i in almost_all_par_id]
                                    if len(intersection) != 0:
                                        print('Two or more parameters have the same Id!')
                                        # sys.exit(1)

                            # get correct observables
                            if data_generator_name == new_species[0]:                                                   # get list of len(measurement) for all parameters + values of one observable
                                correct_string = list_par_id[0]
                                del list_par_id[0]
                                for iObs in list_par_id:
                                    correct_string = correct_string + ';' + iObs
                                for iNumber in range(0, len(time_data)):
                                    new_observables.append(correct_string)
                                new_observables = pd.Series(new_observables)


                # set input
                df_new['observableId'] = new_species
                df_new['measurement'] = new_measurement
                df_new['time'] = time_data

                try:                                                                                                    # short fix
                    df_new['observableParameters'] = new_observables
                except:
                    for iNumber in range(0, len(time_data)):
                        new_observables.append('NaN')
                    new_observables = pd.Series(new_observables)
                    df_new['observableParameters'] = new_observables

                # concatenate data frames
                df = df.append(df_new, ignore_index=True)

            #### save data frame as .tsv
            exp_rearranged_save_path = './sedml2petab/' + sedml_file_name + '/experimental_data_rearranged/' + sedml_file_name + '.tsv'
            df.to_csv(exp_rearranged_save_path, sep='\t', index=False)

    else:
        print(sedml_file_name + ' has no experimental data file!')


    # remove all empty 'experimental_data_rearranged' folders
    if os.path.exists('./sedml2petab/' + sedml_file_name + '/experimental_data_rearranged'):
        all_files = sorted(os.listdir('./sedml2petab/' + sedml_file_name + '/experimental_data_rearranged'))
        if len(all_files) == 0:
            print('The rearrangement did not work!')
            sys.exit()

    return exp_rearranged_save_path