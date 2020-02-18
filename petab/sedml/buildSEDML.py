# create SEDML file with basic properties

import libsedml
import os
import pandas as pd


def createSEDML(sbml_file_name, base_path, exp_con_save_path, new_sbml_file_path):

    # change 'sbml_file_name' to a string without numbers
    new_sbml_file_name = ''.join([i for i in sbml_file_name if not i.isdigit()])

    # create new folder to save the new sedml model
    if not os.path.exists(base_path + '/sedml_file'):
        os.makedirs(base_path + '/sedml_file')

    # create basic structure of the sedml
    sedml_file = libsedml.SedDocument()

    # add 'listOfDataDescription'                                                       ---- linked to 'DataGenerator'
    for iExpDataFile in range(0, len(sorted(os.listdir(base_path + '/experimental_data')))):
        sedml_file.createDataDescription()
        sedml_file.getDataDescription(iExpDataFile).setId(new_sbml_file_name)
        sedml_file.getDataDescription(iExpDataFile).setName(new_sbml_file_name)
        sedml_file.getDataDescription(iExpDataFile).setSource(exp_con_save_path[iExpDataFile])

        # add 'listOfDataSource'                                                        ---- possible error: standard deviations should not be included
        exp_con = pd.read_csv(exp_con_save_path[iExpDataFile])
        header = exp_con.columns
        for iColumn in range(0, len(header)):
            sedml_file.getDataDescription(iExpDataFile).createDataSource()
            sedml_file.getDataDescription(iExpDataFile).getDataSource(iColumn).setId(new_sbml_file_name + '_data_' + header[iColumn])
            sedml_file.getDataDescription(iExpDataFile).getDataSource(iColumn).setIndexSet(header[iColumn])


    # add 'listOfSimulations'
    list_petab_folder = sorted(os.listdir(base_path + '/old_bachmann2011'))
    counter = 0
    for iFile in range(0, len(list_petab_folder)):
        if '.xml' in  list_petab_folder[iFile]:
            counter += counter

    for iSBML in range(0, counter):
        time_course = sedml_file.createUniformTimeCourse()
        time_course.setId('sim' + str(iSBML))
        time_course.setInitialTime(exp_con['time'][0])
        time_course.setOutputStartTime(exp_con['time'][0])
        time_course.setOutputEndTime(exp_con['time'][len(exp_con['time']) - 1])
        time_course.setNumberOfPoints(1000)                                                                             # arbitrary
        ##### what about the algorithm ?


    # add 'listOfModels'
    model = sedml_file.createModel()
    model.setId(sbml_file_name)
    model.setName(sbml_file_name)
    model.setLanguage('urn:sedml:language:sbml')
    model.setSource()


    # add 'listOfTasks'


    # save sedml
    sedml_save_path = base_path + '/sedml_file/' + new_sbml_file_name + '.sedml'
    libsedml.writeSedMLToFile(sedml_file, sedml_save_path)

    return new_sbml_file_name, sedml_save_path