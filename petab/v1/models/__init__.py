"""Handling of different model types supported by PEtab."""

#: SBML model type as used in a PEtab v2 yaml file as `language`.
MODEL_TYPE_SBML = "sbml"
#: PySB model type as used in a PEtab v2 yaml file as `language`.
MODEL_TYPE_PYSB = "pysb"
#: BNGL model type as used in a PEtab v2 yaml file as `language`.
MODEL_TYPE_BNGL = "bngl"

known_model_types = {
    MODEL_TYPE_SBML,
    MODEL_TYPE_PYSB,
    MODEL_TYPE_BNGL,
}

from .model import Model  # noqa F401

__all__ = [
    "MODEL_TYPE_SBML",
    "MODEL_TYPE_PYSB",
    "MODEL_TYPE_BNGL",
    "known_model_types",
    "Model",
]
