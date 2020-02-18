from .sedml_import import *

def downloadAllSBML(sedml_save_path, sbml_save_path):

    sedml_file = libsedml.readSedML(sedml_save_path)

    ######### download all sbml files ###########
    if len(sedml_file.getListOfModels()) == 1:
        sbml_url = sedml_file.getModel(0).getSource()
        sbml_id = sedml_file.getModel(0).getId()
        sbml_save_path = sbml_save_path + '/' + sbml_id + '.sbml'
        download_sbml_model(sbml_url, sbml_save_path)
    else:
        sbml_save_path = sedml_save_path + '/sbml_folder'
        sbml_save_path = download_all_sbml_models_for_sedml_model(sedml_file, sbml_save_path)

    return sbml_save_path, sbml_id