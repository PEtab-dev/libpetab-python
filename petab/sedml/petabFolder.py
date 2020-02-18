# folder containing all necessary petab files

import os
import shutil


def restructureFiles(expconfile_save_path, measdatafile_save_path, parfile_save_path, newest_sbml_save_path, sedml_file_name):

    # create new folder for all files
    final_folder_path = './sedml2petab/' + sedml_file_name + '/' + sedml_file_name
    if not os.path.exists(final_folder_path):
        os.makedirs(final_folder_path)

    # copy all necessary files into the new folder
    shutil.copyfile(expconfile_save_path, final_folder_path + '/experimentalCondition_' + sedml_file_name + '.tsv')
    shutil.copyfile(measdatafile_save_path, final_folder_path + '/measurementData_' + sedml_file_name + '.tsv')
    shutil.copyfile(parfile_save_path, final_folder_path + '/parameters_' + sedml_file_name + '.tsv')
    shutil.copyfile(newest_sbml_save_path, final_folder_path + '/model_' + sedml_file_name + '.xml')


    return final_folder_path