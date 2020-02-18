from .sedml_import import *
import os

def downloadAllSEDML(sedml_path, sedml_file_name):

    ########### create folder for all sedml and sbml files ##############
    if not os.path.exists('./sedml2petab'):
        os.makedirs('./sedml2petab')
    if not os.path.exists('./sedml2petab/' + sedml_file_name):
        os.makedirs('./sedml2petab/' + sedml_file_name)
    if not os.path.exists('./sedml2petab/' + sedml_file_name + '/sbml_models'):
        os.makedirs('./sedml2petab/' + sedml_file_name + '/sbml_models')

    sedml_save_path = './sedml2petab'
    sbml_save_path = './sedml2petab/' + sedml_file_name + '/sbml_models'

    if not os.path.isfile(sedml_path):
        ############## download sedml + open sedml + download sbml + open sbml ################
        try:
            download_sedml_model(sedml_path, sedml_save_path)
        except:
            raise ValueError('The sedml path is not a file nor a valid URL!')

    else:
        ############## copy sedml + download sbml + open sbml #################
        shutil.copyfile(sedml_path, sedml_save_path + '/' + sedml_file_name + '/' + sedml_file_name + '.sedml')
        sedml_save_path = sedml_save_path + '/' + sedml_file_name + '/' + sedml_file_name + '.sedml'


    return sedml_save_path, sbml_save_path
