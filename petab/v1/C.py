# pylint: disable:invalid-name
"""
This file contains constant definitions.
"""
import math as _math
import sys

# MEASUREMENTS

#: Observable ID column in the observable and measurement tables
OBSERVABLE_ID = "observableId"

#: Preequilibration condition ID column in the measurement table
PREEQUILIBRATION_CONDITION_ID = "preequilibrationConditionId"

#: Simulation condition ID column in the measurement table
SIMULATION_CONDITION_ID = "simulationConditionId"

#: Measurement value column in the measurement table
MEASUREMENT = "measurement"

#: Time column in the measurement table
TIME = "time"

#: Time value that indicates steady-state measurements
TIME_STEADY_STATE = _math.inf

#: Observable parameters column in the measurement table
OBSERVABLE_PARAMETERS = "observableParameters"

#: Noise parameters column in the measurement table
NOISE_PARAMETERS = "noiseParameters"

#: Dataset ID column in the measurement table
DATASET_ID = "datasetId"

#: Replicate ID column in the measurement table
REPLICATE_ID = "replicateId"

#: Mandatory columns of measurement table
MEASUREMENT_DF_REQUIRED_COLS = [
    OBSERVABLE_ID,
    SIMULATION_CONDITION_ID,
    MEASUREMENT,
    TIME,
]

#: Optional columns of measurement table
MEASUREMENT_DF_OPTIONAL_COLS = [
    PREEQUILIBRATION_CONDITION_ID,
    OBSERVABLE_PARAMETERS,
    NOISE_PARAMETERS,
    DATASET_ID,
    REPLICATE_ID,
]

#: Measurement table columns
MEASUREMENT_DF_COLS = [
    MEASUREMENT_DF_REQUIRED_COLS[0],
    MEASUREMENT_DF_OPTIONAL_COLS[0],
    *MEASUREMENT_DF_REQUIRED_COLS[1:],
    *MEASUREMENT_DF_OPTIONAL_COLS[1:],
]


# PARAMETERS

#: Parameter ID column in the parameter table
PARAMETER_ID = "parameterId"
#: Parameter name column in the parameter table
PARAMETER_NAME = "parameterName"
#: Parameter scale column in the parameter table
PARAMETER_SCALE = "parameterScale"
#: Lower bound column in the parameter table
LOWER_BOUND = "lowerBound"
#: Upper bound column in the parameter table
UPPER_BOUND = "upperBound"
#: Nominal value column in the parameter table
NOMINAL_VALUE = "nominalValue"
#: Estimate column in the parameter table
ESTIMATE = "estimate"
#: Initialization prior type column in the parameter table
INITIALIZATION_PRIOR_TYPE = "initializationPriorType"
#: Initialization prior parameters column in the parameter table
INITIALIZATION_PRIOR_PARAMETERS = "initializationPriorParameters"
#: Objective prior type column in the parameter table
OBJECTIVE_PRIOR_TYPE = "objectivePriorType"
#: Objective prior parameters column in the parameter table
OBJECTIVE_PRIOR_PARAMETERS = "objectivePriorParameters"

#: Mandatory columns of parameter table
PARAMETER_DF_REQUIRED_COLS = [
    PARAMETER_ID,
    PARAMETER_SCALE,
    LOWER_BOUND,
    UPPER_BOUND,
    ESTIMATE,
]

#: Optional columns of parameter table
PARAMETER_DF_OPTIONAL_COLS = [
    PARAMETER_NAME,
    NOMINAL_VALUE,
    INITIALIZATION_PRIOR_TYPE,
    INITIALIZATION_PRIOR_PARAMETERS,
    OBJECTIVE_PRIOR_TYPE,
    OBJECTIVE_PRIOR_PARAMETERS,
]

#: Parameter table columns
PARAMETER_DF_COLS = [
    PARAMETER_DF_REQUIRED_COLS[0],
    PARAMETER_DF_OPTIONAL_COLS[0],
    *PARAMETER_DF_REQUIRED_COLS[1:],
    *PARAMETER_DF_OPTIONAL_COLS[1:],
]

#: Initialization-type prior
INITIALIZATION = "initialization"
#: Objective-type prior
OBJECTIVE = "objective"


# CONDITIONS

#: Condition ID column in the condition table
CONDITION_ID = "conditionId"
#: Condition name column in the condition table
CONDITION_NAME = "conditionName"


# OBSERVABLES

#: Observable name column in the observables table
OBSERVABLE_NAME = "observableName"
#: Observable formula column in the observables table
OBSERVABLE_FORMULA = "observableFormula"
#: Noise formula column in the observables table
NOISE_FORMULA = "noiseFormula"
#: Observable transformation column in the observables table
OBSERVABLE_TRANSFORMATION = "observableTransformation"
#: Noise distribution column in the observables table
NOISE_DISTRIBUTION = "noiseDistribution"

#: Mandatory columns of observables table
OBSERVABLE_DF_REQUIRED_COLS = [
    OBSERVABLE_ID,
    OBSERVABLE_FORMULA,
    NOISE_FORMULA,
]

#: Optional columns of observables table
OBSERVABLE_DF_OPTIONAL_COLS = [
    OBSERVABLE_NAME,
    OBSERVABLE_TRANSFORMATION,
    NOISE_DISTRIBUTION,
]

#: Observables table columns
OBSERVABLE_DF_COLS = [
    *OBSERVABLE_DF_REQUIRED_COLS,
    *OBSERVABLE_DF_OPTIONAL_COLS,
]


# TRANSFORMATIONS

#: Linear transformation
LIN = "lin"
#: Logarithmic transformation
LOG = "log"
#: Logarithmic base 10 transformation
LOG10 = "log10"
#: Supported observable transformations
OBSERVABLE_TRANSFORMATIONS = [LIN, LOG, LOG10]


# NOISE MODELS

#: Uniform distribution
UNIFORM = "uniform"
#: Uniform distribution on the parameter scale
PARAMETER_SCALE_UNIFORM = "parameterScaleUniform"
#: Normal distribution
NORMAL = "normal"
#: Normal distribution on the parameter scale
PARAMETER_SCALE_NORMAL = "parameterScaleNormal"
#: Laplace distribution
LAPLACE = "laplace"
#: Laplace distribution on the parameter scale
PARAMETER_SCALE_LAPLACE = "parameterScaleLaplace"
#: Log-normal distribution
LOG_NORMAL = "logNormal"
#: Log-Laplace distribution
LOG_LAPLACE = "logLaplace"

#: Supported prior types
PRIOR_TYPES = [
    UNIFORM,
    NORMAL,
    LAPLACE,
    LOG_NORMAL,
    LOG_LAPLACE,
    PARAMETER_SCALE_UNIFORM,
    PARAMETER_SCALE_NORMAL,
    PARAMETER_SCALE_LAPLACE,
]

#: Supported noise distributions
NOISE_MODELS = [NORMAL, LAPLACE]


# VISUALIZATION

#: Plot ID column in the visualization table
PLOT_ID = "plotId"
#: Plot name column in the visualization table
PLOT_NAME = "plotName"
#: Value for plot type 'simulation' in the visualization table
PLOT_TYPE_SIMULATION = "plotTypeSimulation"
#: Value for plot type 'data' in the visualization table
PLOT_TYPE_DATA = "plotTypeData"
#: X values column in the visualization table
X_VALUES = "xValues"
#: X offset column in the visualization table
X_OFFSET = "xOffset"
#: X label column in the visualization table
X_LABEL = "xLabel"
#: X scale column in the visualization table
X_SCALE = "xScale"
#: Y values column in the visualization table
Y_VALUES = "yValues"
#: Y offset column in the visualization table
Y_OFFSET = "yOffset"
#: Y label column in the visualization table
Y_LABEL = "yLabel"
#: Y scale column in the visualization table
Y_SCALE = "yScale"
#: Legend entry column in the visualization table
LEGEND_ENTRY = "legendEntry"

#: Mandatory columns of visualization table
VISUALIZATION_DF_REQUIRED_COLS = [PLOT_ID]

#: Optional columns of visualization table
VISUALIZATION_DF_OPTIONAL_COLS = [
    PLOT_NAME,
    PLOT_TYPE_SIMULATION,
    PLOT_TYPE_DATA,
    X_VALUES,
    X_OFFSET,
    X_LABEL,
    X_SCALE,
    Y_VALUES,
    Y_OFFSET,
    Y_LABEL,
    Y_SCALE,
    LEGEND_ENTRY,
    DATASET_ID,
]

#: Visualization table columns
VISUALIZATION_DF_COLS = [
    *VISUALIZATION_DF_REQUIRED_COLS,
    *VISUALIZATION_DF_OPTIONAL_COLS,
]

#: Visualization table columns that contain subplot specifications
VISUALIZATION_DF_SUBPLOT_LEVEL_COLS = [
    PLOT_ID,
    PLOT_NAME,
    PLOT_TYPE_SIMULATION,
    PLOT_TYPE_DATA,
    X_LABEL,
    X_SCALE,
    Y_LABEL,
    Y_SCALE,
]

#: Visualization table columns that contain single plot specifications
VISUALIZATION_DF_SINGLE_PLOT_LEVEL_COLS = [
    X_VALUES,
    X_OFFSET,
    Y_VALUES,
    Y_OFFSET,
    LEGEND_ENTRY,
    DATASET_ID,
]

#: Plot type value in the visualization table for line plot
LINE_PLOT = "LinePlot"
#: Plot type value in the visualization table for bar plot
BAR_PLOT = "BarPlot"
#: Plot type value in the visualization table for scatter plot
SCATTER_PLOT = "ScatterPlot"
#: Supported plot types
PLOT_TYPES_SIMULATION = [LINE_PLOT, BAR_PLOT, SCATTER_PLOT]

#: Supported xScales
X_SCALES = [LIN, LOG, LOG10]

#: Supported yScales
Y_SCALES = [LIN, LOG, LOG10]


#: Plot type "data" value in the visualization table for mean and standard
#  deviation
MEAN_AND_SD = "MeanAndSD"
#: Plot type "data" value in the visualization table for mean and standard
#  error
MEAN_AND_SEM = "MeanAndSEM"
#: Plot type "data" value in the visualization table for replicates
REPLICATE = "replicate"
#: Plot type "data" value in the visualization table for provided noise values
PROVIDED = "provided"
#: Supported settings for handling replicates
PLOT_TYPES_DATA = [MEAN_AND_SD, MEAN_AND_SEM, REPLICATE, PROVIDED]


# YAML
#: PEtab version key in the YAML file
FORMAT_VERSION = "format_version"
#: Parameter file key in the YAML file
PARAMETER_FILE = "parameter_file"
#: Problems key in the YAML file
PROBLEMS = "problems"
#: SBML files key in the YAML file
SBML_FILES = "sbml_files"
#: Model files key in the YAML file
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
MODEL_FILES = "model_files"
#: Model location key in the YAML file
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
MODEL_LOCATION = "location"
#: Model language key in the YAML file
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
MODEL_LANGUAGE = "language"
#: Condition files key in the YAML file
CONDITION_FILES = "condition_files"
#: Measurement files key in the YAML file
MEASUREMENT_FILES = "measurement_files"
#: Observable files key in the YAML file
OBSERVABLE_FILES = "observable_files"
#: Visualization files key in the YAML file
VISUALIZATION_FILES = "visualization_files"
#: Mapping files key in the YAML file
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
MAPPING_FILES = "mapping_files"
#: Extensions key in the YAML file
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
EXTENSIONS = "extensions"


# MAPPING

#: PEtab entity ID column in the mapping table
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
PETAB_ENTITY_ID = "petabEntityId"
#: Model entity ID column in the mapping table
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
MODEL_ENTITY_ID = "modelEntityId"
#: Required columns of the mapping table
#  (PEtab v2.0 -- DEPRECATED: use value from petab.v2.C)
MAPPING_DF_REQUIRED_COLS = [PETAB_ENTITY_ID, MODEL_ENTITY_ID]

# MORE

#: Simulated value column in the simulation table
SIMULATION = "simulation"
#: Residual value column in the residuals table
RESIDUAL = "residual"
#: ???
NOISE_VALUE = "noiseValue"

#: separator for multiple parameter values (bounds, observableParameters, ...)
PARAMETER_SEPARATOR = ";"


__all__ = [
    x
    for x in dir(sys.modules[__name__])
    if not x.startswith("_") and x not in {"sys", "math"}
]
