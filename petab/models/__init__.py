MODEL_TYPE_SBML = 'sbml'
MODEL_TYPE_PYSB = 'pysb'

known_model_types = {
    MODEL_TYPE_SBML,
    MODEL_TYPE_PYSB,
}

from .model import Model  # noqa F401
