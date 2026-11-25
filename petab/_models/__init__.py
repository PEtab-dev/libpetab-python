"""Model types supported by PEtab, shared by :mod:`petab.v1` and
:mod:`petab.v2`.

Use :mod:`petab.v1.models` or :mod:`petab.v2.models` instead of importing
from here.
"""

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

# The model modules use helpers from `petab.v1`, whose initialization
#  imports `petab.v1.models` and thus this package; initialize `petab.v1`
#  first to avoid a circular import.
import petab.v1  # noqa: E402, F401

from .model import Model  # noqa: E402

__all__ = [
    "MODEL_TYPE_BNGL",
    "MODEL_TYPE_PYSB",
    "MODEL_TYPE_SBML",
    "Model",
    "known_model_types",
]
