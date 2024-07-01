"""Handling of different model types supported by PEtab."""
#: SBML model type as used in a PEtab v2 yaml file as `language`.
MODEL_TYPE_SBML = "sbml"
#: PySB model type as used in a PEtab v2 yaml file as `language`.
MODEL_TYPE_PYSB = "pysb"

known_model_types = {
    MODEL_TYPE_SBML,
    MODEL_TYPE_PYSB,
}

from .model import Model  # noqa F401

__all__ = ["MODEL_TYPE_SBML", "MODEL_TYPE_PYSB", "known_model_types", "Model"]
