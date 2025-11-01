# pylint: disable:invalid-name
"""
This file contains constant definitions.
"""

import math as _math
import sys

# MEASUREMENTS

#: Observable ID column in the observable and measurement tables
OBSERVABLE_ID = "observableId"

#: Experiment ID column in the measurement table
EXPERIMENT_ID = "experimentId"

#: Measurement value column in the measurement table
MEASUREMENT = "measurement"

#: Time column in the measurement table
TIME = "time"

#: Time value that indicates steady-state measurements
TIME_STEADY_STATE = _math.inf

#: Time value that indicates pre-equilibration in the experiments table
TIME_PREEQUILIBRATION = -_math.inf

#: Observable parameters column in the measurement table
OBSERVABLE_PARAMETERS = "observableParameters"

#: Noise parameters column in the measurement table
NOISE_PARAMETERS = "noiseParameters"

#: Dataset ID column in the measurement table
DATASET_ID = "datasetId"

#: Replicate ID column in the measurement table
REPLICATE_ID = "replicateId"

#: The model ID column in the measurement table
MODEL_ID = "modelId"

#: Mandatory columns of measurement table
MEASUREMENT_DF_REQUIRED_COLS = [
    OBSERVABLE_ID,
    EXPERIMENT_ID,
    MEASUREMENT,
    TIME,
]

#: Optional columns of measurement table
MEASUREMENT_DF_OPTIONAL_COLS = [
    OBSERVABLE_PARAMETERS,
    NOISE_PARAMETERS,
    DATASET_ID,
    REPLICATE_ID,
    MODEL_ID,
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
#: Lower bound column in the parameter table
LOWER_BOUND = "lowerBound"
#: Upper bound column in the parameter table
UPPER_BOUND = "upperBound"
#: Nominal value column in the parameter table
NOMINAL_VALUE = "nominalValue"
#: Estimate column in the parameter table
ESTIMATE = "estimate"
#: Prior distribution type column in the parameter table
PRIOR_DISTRIBUTION = "priorDistribution"
#: Prior parameters column in the parameter table
PRIOR_PARAMETERS = "priorParameters"

#: Mandatory columns of parameter table
PARAMETER_DF_REQUIRED_COLS = [
    PARAMETER_ID,
    LOWER_BOUND,
    UPPER_BOUND,
    ESTIMATE,
]

#: Optional columns of parameter table
PARAMETER_DF_OPTIONAL_COLS = [
    PARAMETER_NAME,
    NOMINAL_VALUE,
    PRIOR_DISTRIBUTION,
    PRIOR_PARAMETERS,
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
#: Column in the condition table with the ID of an entity that is changed
TARGET_ID = "targetId"
#: Column in the condition table with the new value of the target entity
TARGET_VALUE = "targetValue"

CONDITION_DF_COLS = [
    CONDITION_ID,
    TARGET_ID,
    TARGET_VALUE,
]

CONDITION_DF_REQUIRED_COLS = CONDITION_DF_COLS

# EXPERIMENTS
EXPERIMENT_DF_REQUIRED_COLS = [
    EXPERIMENT_ID,
    TIME,
    CONDITION_ID,
]

# OBSERVABLES

#: Observable name column in the observable table
OBSERVABLE_NAME = "observableName"
#: Observable formula column in the observable table
OBSERVABLE_FORMULA = "observableFormula"
#: Observable placeholders column in the observable table
OBSERVABLE_PLACEHOLDERS = "observablePlaceholders"
#: Noise formula column in the observable table
NOISE_FORMULA = "noiseFormula"
#: Noise distribution column in the observable table
NOISE_DISTRIBUTION = "noiseDistribution"
#: Noise placeholders column in the observable table
NOISE_PLACEHOLDERS = "noisePlaceholders"

#: Mandatory columns of observable table
OBSERVABLE_DF_REQUIRED_COLS = [
    OBSERVABLE_ID,
    OBSERVABLE_FORMULA,
    NOISE_FORMULA,
]

#: Optional columns of observable table
OBSERVABLE_DF_OPTIONAL_COLS = [
    OBSERVABLE_NAME,
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


# NOISE MODELS


#: Cauchy distribution.
CAUCHY = "cauchy"
#: Chi-squared distribution.
# FIXME: "chisquare" in PEtab and sbml-distrib, but usually "chi-squared"
CHI_SQUARED = "chisquare"
#: Exponential distribution.
EXPONENTIAL = "exponential"
#: Gamma distribution.
GAMMA = "gamma"
#: Laplace distribution
LAPLACE = "laplace"
#: Log-Laplace distribution
LOG_LAPLACE = "log-laplace"
#: Log-normal distribution
LOG_NORMAL = "log-normal"
#: Log-uniform distribution.
LOG_UNIFORM = "log-uniform"
#: Normal distribution
NORMAL = "normal"
#: Rayleigh distribution.
RAYLEIGH = "rayleigh"
#: Uniform distribution
UNIFORM = "uniform"

#: Supported prior distribution types
PRIOR_DISTRIBUTIONS = [
    CAUCHY,
    CHI_SQUARED,
    EXPONENTIAL,
    GAMMA,
    LAPLACE,
    LOG_LAPLACE,
    LOG_NORMAL,
    LOG_UNIFORM,
    NORMAL,
    RAYLEIGH,
    UNIFORM,
]


#: Supported noise distributions
NOISE_DISTRIBUTIONS = [NORMAL, LAPLACE, LOG_NORMAL, LOG_LAPLACE]

# YAML
#: PEtab version key in the YAML file
FORMAT_VERSION = "format_version"
#: Parameter files key in the YAML file
PARAMETER_FILES = "parameter_files"
#: Problems key in the YAML file
PROBLEMS = "problems"
#: Model files key in the YAML file
MODEL_FILES = "model_files"
#: Model location key in the YAML file
MODEL_LOCATION = "location"
#: Model language key in the YAML file
MODEL_LANGUAGE = "language"
#: Condition files key in the YAML file
CONDITION_FILES = "condition_files"
#: Experiment files key in the YAML file
EXPERIMENT_FILES = "experiment_files"
#: Measurement files key in the YAML file
MEASUREMENT_FILES = "measurement_files"
#: Observable files key in the YAML file
OBSERVABLE_FILES = "observable_files"
#: Visualization files key in the YAML file
VISUALIZATION_FILES = "visualization_files"
#: Mapping files key in the YAML file
MAPPING_FILES = "mapping_files"
#: Extensions key in the YAML file
EXTENSIONS = "extensions"


# MAPPING

#: PEtab entity ID column in the mapping table
PETAB_ENTITY_ID = "petabEntityId"
#: Model entity ID column in the mapping table
MODEL_ENTITY_ID = "modelEntityId"
#: Arbitrary name
NAME = "name"

#: Required columns of the mapping table
MAPPING_DF_REQUIRED_COLS = [PETAB_ENTITY_ID, MODEL_ENTITY_ID]

# MORE

#: Simulated value column in the simulation table
SIMULATION = "simulation"
#: Residual value column in the residual table
RESIDUAL = "residual"

#: separator for multiple parameter values (bounds, observableParameters, ...)
PARAMETER_SEPARATOR = ";"

#: The time symbol for use in any PEtab-specific mathematical expressions
TIME_SYMBOL = "time"


__all__ = [
    x
    for x in dir(sys.modules[__name__])
    if not x.startswith("_") and x not in {"sys", "math"}
]
