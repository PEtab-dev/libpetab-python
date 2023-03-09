# pylint: disable:invalid-name
"""
This file contains constant definitions.
"""

import math as _math


# MEASUREMENTS

#:
OBSERVABLE_ID = 'observableId'

#:
PREEQUILIBRATION_CONDITION_ID = 'preequilibrationConditionId'

#:
SIMULATION_CONDITION_ID = 'simulationConditionId'

#:
MEASUREMENT = 'measurement'

#:
TIME = 'time'

#: Time value that indicates steady-state measurements
TIME_STEADY_STATE = _math.inf

#:
OBSERVABLE_PARAMETERS = 'observableParameters'

#:
NOISE_PARAMETERS = 'noiseParameters'

#:
DATASET_ID = 'datasetId'

#:
REPLICATE_ID = 'replicateId'

#: Mandatory columns of measurement table
MEASUREMENT_DF_REQUIRED_COLS = [
    OBSERVABLE_ID, SIMULATION_CONDITION_ID, MEASUREMENT, TIME]

#: Optional columns of measurement table
MEASUREMENT_DF_OPTIONAL_COLS = [
    PREEQUILIBRATION_CONDITION_ID, OBSERVABLE_PARAMETERS,
    NOISE_PARAMETERS,
    DATASET_ID, REPLICATE_ID]

#: Measurement table columns
MEASUREMENT_DF_COLS = [
    MEASUREMENT_DF_REQUIRED_COLS[0], MEASUREMENT_DF_OPTIONAL_COLS[0],
    *MEASUREMENT_DF_REQUIRED_COLS[1:], *MEASUREMENT_DF_OPTIONAL_COLS[1:]]


# PARAMETERS

#:
PARAMETER_ID = 'parameterId'
#:
PARAMETER_NAME = 'parameterName'
#:
PARAMETER_SCALE = 'parameterScale'
#:
LOWER_BOUND = 'lowerBound'
#:
UPPER_BOUND = 'upperBound'
#:
NOMINAL_VALUE = 'nominalValue'
#:
ESTIMATE = 'estimate'
#:
INITIALIZATION_PRIOR_TYPE = 'initializationPriorType'
#:
INITIALIZATION_PRIOR_PARAMETERS = 'initializationPriorParameters'
#:
OBJECTIVE_PRIOR_TYPE = 'objectivePriorType'
#:
OBJECTIVE_PRIOR_PARAMETERS = 'objectivePriorParameters'

#: Mandatory columns of parameter table
PARAMETER_DF_REQUIRED_COLS = [
    PARAMETER_ID, PARAMETER_SCALE, LOWER_BOUND, UPPER_BOUND, ESTIMATE]

#: Optional columns of parameter table
PARAMETER_DF_OPTIONAL_COLS = [
    PARAMETER_NAME, NOMINAL_VALUE,
    INITIALIZATION_PRIOR_TYPE, INITIALIZATION_PRIOR_PARAMETERS,
    OBJECTIVE_PRIOR_TYPE, OBJECTIVE_PRIOR_PARAMETERS]

#: Parameter table columns
PARAMETER_DF_COLS = [
    PARAMETER_DF_REQUIRED_COLS[0], PARAMETER_DF_OPTIONAL_COLS[0],
    *PARAMETER_DF_REQUIRED_COLS[1:], *PARAMETER_DF_OPTIONAL_COLS[1:]]

#:
INITIALIZATION = 'initialization'
#:
OBJECTIVE = 'objective'


# CONDITIONS

#:
CONDITION_ID = 'conditionId'
#:
CONDITION_NAME = 'conditionName'


# OBSERVABLES

#:
OBSERVABLE_NAME = 'observableName'
#:
OBSERVABLE_FORMULA = 'observableFormula'
#:
NOISE_FORMULA = 'noiseFormula'
#:
OBSERVABLE_TRANSFORMATION = 'observableTransformation'
#:
NOISE_DISTRIBUTION = 'noiseDistribution'

#: Mandatory columns of observables table
OBSERVABLE_DF_REQUIRED_COLS = [
    OBSERVABLE_ID, OBSERVABLE_FORMULA, NOISE_FORMULA]

#: Optional columns of observables table
OBSERVABLE_DF_OPTIONAL_COLS = [
    OBSERVABLE_NAME, OBSERVABLE_TRANSFORMATION, NOISE_DISTRIBUTION]

#: Observables table columns
OBSERVABLE_DF_COLS = [
    *OBSERVABLE_DF_REQUIRED_COLS, *OBSERVABLE_DF_OPTIONAL_COLS]


# TRANSFORMATIONS

#:
LIN = 'lin'
#:
LOG = 'log'
#:
LOG10 = 'log10'
#: Supported observable transformations
OBSERVABLE_TRANSFORMATIONS = [LIN, LOG, LOG10]


# NOISE MODELS

#:
UNIFORM = 'uniform'
#:
PARAMETER_SCALE_UNIFORM = 'parameterScaleUniform'
#:
NORMAL = 'normal'
#:
PARAMETER_SCALE_NORMAL = 'parameterScaleNormal'
#:
LAPLACE = 'laplace'
#:
PARAMETER_SCALE_LAPLACE = 'parameterScaleLaplace'
#:
LOG_NORMAL = 'logNormal'
#:
LOG_LAPLACE = 'logLaplace'

#: Supported prior types
PRIOR_TYPES = [
    UNIFORM, NORMAL, LAPLACE, LOG_NORMAL, LOG_LAPLACE,
    PARAMETER_SCALE_UNIFORM, PARAMETER_SCALE_NORMAL, PARAMETER_SCALE_LAPLACE]

#: Supported noise distributions
NOISE_MODELS = [NORMAL, LAPLACE]


# VISUALIZATION

#:
PLOT_ID = 'plotId'
#:
PLOT_NAME = 'plotName'
#:
PLOT_TYPE_SIMULATION = 'plotTypeSimulation'
#:
PLOT_TYPE_DATA = 'plotTypeData'
#:
X_VALUES = 'xValues'
#:
X_OFFSET = 'xOffset'
#:
X_LABEL = 'xLabel'
#:
X_SCALE = 'xScale'
#:
Y_VALUES = 'yValues'
#:
Y_OFFSET = 'yOffset'
#:
Y_LABEL = 'yLabel'
#:
Y_SCALE = 'yScale'
#:
LEGEND_ENTRY = 'legendEntry'

#: Mandatory columns of visualization table
VISUALIZATION_DF_REQUIRED_COLS = [PLOT_ID]

#: Optional columns of visualization table
VISUALIZATION_DF_OPTIONAL_COLS = [
    PLOT_NAME, PLOT_TYPE_SIMULATION, PLOT_TYPE_DATA, X_VALUES, X_OFFSET,
    X_LABEL, X_SCALE, Y_VALUES, Y_OFFSET, Y_LABEL, Y_SCALE, LEGEND_ENTRY,
    DATASET_ID]

#: Visualization table columns
VISUALIZATION_DF_COLS = [
    *VISUALIZATION_DF_REQUIRED_COLS, *VISUALIZATION_DF_OPTIONAL_COLS]

#: Visualization table columns that contain subplot specifications
VISUALIZATION_DF_SUBPLOT_LEVEL_COLS = [
    PLOT_ID, PLOT_NAME, PLOT_TYPE_SIMULATION, PLOT_TYPE_DATA,
    X_LABEL, X_SCALE, Y_LABEL, Y_SCALE]

#: Visualization table columns that contain single plot specifications
VISUALIZATION_DF_SINGLE_PLOT_LEVEL_COLS = [
    X_VALUES, X_OFFSET, Y_VALUES, Y_OFFSET, LEGEND_ENTRY, DATASET_ID]

#:
LINE_PLOT = 'LinePlot'
#:
BAR_PLOT = 'BarPlot'
#:
SCATTER_PLOT = 'ScatterPlot'
#: Supported plot types
PLOT_TYPES_SIMULATION = [LINE_PLOT, BAR_PLOT, SCATTER_PLOT]

#: Supported xScales
X_SCALES = [LIN, LOG, LOG10]

#: Supported yScales
Y_SCALES = [LIN, LOG, LOG10]


#:
MEAN_AND_SD = 'MeanAndSD'
#:
MEAN_AND_SEM = 'MeanAndSEM'
#:
REPLICATE = 'replicate'
#:
PROVIDED = 'provided'
#: Supported settings for handling replicates
PLOT_TYPES_DATA = [MEAN_AND_SD, MEAN_AND_SEM, REPLICATE, PROVIDED]


# YAML
#:
FORMAT_VERSION = 'format_version'
#:
PARAMETER_FILE = 'parameter_file'
#:
PROBLEMS = 'problems'
#:
SBML_FILES = 'sbml_files'
#:
MODEL_FILES = 'model_files'
#:
MODEL_LOCATION = 'location'
#:
MODEL_LANGUAGE = 'language'
#:
CONDITION_FILES = 'condition_files'
#:
MEASUREMENT_FILES = 'measurement_files'
#:
OBSERVABLE_FILES = 'observable_files'
#:
VISUALIZATION_FILES = 'visualization_files'
#:
MAPPING_FILES = 'mapping_files'
#:
EXTENSIONS = 'extensions'


# MAPPING
#:
PETAB_ENTITY_ID = 'petabEntityId'
#:
MODEL_ENTITY_ID = 'modelEntityId'
#:
MAPPING_DF_REQUIRED_COLS = [PETAB_ENTITY_ID, MODEL_ENTITY_ID]

# MORE

#:
SIMULATION = 'simulation'
#:
RESIDUAL = 'residual'
#:
NOISE_VALUE = 'noiseValue'

# separator for multiple parameter values (bounds, observableParameters, ...)
PARAMETER_SEPARATOR = ';'
