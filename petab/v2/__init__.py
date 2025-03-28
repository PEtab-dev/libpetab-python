"""The PEtab 2.0 subpackage.

Contains all functionality related to handling PEtab 2.0 problems.
"""

from warnings import warn

# TODO: remove v1 star imports
from ..v1.calculate import *  # noqa: F403, F401, E402
from ..v1.composite_problem import *  # noqa: F403, F401, E402
from ..v1.core import *  # noqa: F403, F401, E402
from ..v1.format_version import __format_version__  # noqa: F401, E402
from ..v1.mapping import *  # noqa: F403, F401, E402
from ..v1.measurements import *  # noqa: F403, F401, E402
from ..v1.observables import *  # noqa: F403, F401, E402
from ..v1.parameter_mapping import *  # noqa: F403, F401, E402
from ..v1.parameters import *  # noqa: F403, F401, E402
from ..v1.sampling import *  # noqa: F403, F401, E402
from ..v1.sbml import *  # noqa: F403, F401, E402
from ..v1.simulate import *  # noqa: F403, F401, E402
from ..v1.yaml import *  # noqa: F403, F401, E402

warn(
    "Support for PEtab2.0 and all of petab.v2 is experimental "
    "and subject to changes!",
    stacklevel=1,
)

# import after v1
from ..version import __version__  # noqa: F401, E402
from . import (  # noqa: F401, E402
    C,  # noqa: F401, E402
    models,  # noqa: F401, E402
)
from .conditions import *  # noqa: F403, F401, E402
from .experiments import (  # noqa: F401, E402
    get_experiment_df,
    write_experiment_df,
)
from .lint import lint_problem  # noqa: F401, E402
from .models import Model  # noqa: F401, E402
from .problem import Problem  # noqa: F401, E402
