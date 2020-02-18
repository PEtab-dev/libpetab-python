# change the observable Id from 'getObservables.py' to the new default of the petab format

def changeObsId(sbml_file, sigma_parameters):

    # delete parameter that starts with 'observable_' to avoid ambiguity
    sbml_model = sbml_file.getModel()
    delete_counter = 0
    for iPar in range(0, sbml_model.getNumParameters()):
        if 'observable_' in sbml_model.getParameter(iPar - delete_counter).getId():
            sbml_model.removeParameter(iPar - delete_counter)
            delete_counter += 1

    # rename the assignment rules to the petab default
    for iAssignmentRule in range(0, sbml_model.getNumRules()):
        for iNewDefault in range(0, len(sigma_parameters)):
            if sigma_parameters[iNewDefault].split('sigma')[1] in sbml_model.getRule(iAssignmentRule).getVariable() or \
                    sigma_parameters[iNewDefault].split('sigma')[1].lower() in sbml_model.getRule(iAssignmentRule).getVariable() or \
                    sigma_parameters[iNewDefault].split('sigma')[1].upper() in sbml_model.getRule(iAssignmentRule).getVariable():
                sbml_model.getRule(iAssignmentRule).setVariable('observable_' + sigma_parameters[iNewDefault].split('sigma_')[1])
                #sbml_model.getRule(iAssignmentRule).getVariable().replace(sbml_model.getRule(iAssignmentRule).getVariable(), 'observable_' + sigma_parameters[iNewDefault].split('sigma_')[1])
                #sbml_model.getRule(iAssignmentRule).getVariable() = 'observable_' + sigma_parameters[iNewDefault].split('sigma_')[1]
                break


    return sbml_file