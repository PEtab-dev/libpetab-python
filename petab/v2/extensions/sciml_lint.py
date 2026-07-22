"""Validation tasks for the PEtab SciML extension."""

from __future__ import annotations

import math

from petab_sciml import (
    ALL_CONDITION_IDS,
    ARRAY,
    NN_ENTITY_PATTERN,
    NN_PARAMETER_PATTERN,
    ActivationFunctions,
    Layers,
    Op,
    TensorOps,
)

from petab.v1.lint import is_valid_identifier
from petab.v1.math import sympify_petab

from .. import core, lint
from ..C import EXT_ID_SCIML

__all__ = [
    "CheckNeuralNetworkModel",
    "CheckHybridizationTable",
    "CheckSciMLConditionTable",
    "CheckArrayDataFiles",
    "CheckSciMLParameterTable",
]

#: Placeholder used in messages when a neural network has no ID.
MISSING_NN_ID = "MISSING ID"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sciml_config(problem: core.Problem):
    """Return the :class:`SciMLConfig` from the problem config, if any."""
    if problem.config is None or not problem.config.extensions:
        return None
    return problem.config.extensions.get(EXT_ID_SCIML)


def _nn_models(problem: core.Problem) -> list:
    """The loaded (e.g. YAML-format) neural network models."""
    return [
        nn
        for nn in problem.extensions.sciml.neural_networks
        # Only YAML-format networks are loaded with introspectable layers.
        if hasattr(nn, "layers")
    ]


def _nn_ids(problem: core.Problem) -> set[str]:
    """All neural network IDs referenced by the problem."""
    ids = {
        nn.nn_model_id
        for nn in problem.extensions.sciml.neural_networks
        if hasattr(nn, "nn_model_id")
    }
    if (config := _sciml_config(problem)) is not None:
        ids |= set(config.neural_networks or {})
    return ids


def _nn_entity_petab_ids(
    problem: core.Problem,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Classify NN entities referenced in the mapping table.

    Returns three dicts (inputs, outputs, parameters), each mapping the PEtab
    entity ID to the neural network ID it belongs to.
    """
    nn_ids = _nn_ids(problem)
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    parameters: dict[str, str] = {}
    buckets = {"inputs": inputs, "outputs": outputs, "parameters": parameters}
    for mapping in problem.mappings:
        if not mapping.model_id:
            continue
        if not (match := NN_ENTITY_PATTERN.match(mapping.model_id)):
            continue
        # Only consider references to actual neural networks.
        if nn_ids and match.group("nn_id") not in nn_ids:
            continue
        buckets[match.group("kind")][mapping.petab_id] = match.group("nn_id")
    return inputs, outputs, parameters


def _array_input_ids(problem: core.Problem) -> set[str]:
    """Input IDs that have array data defined across all array data files."""
    return {
        input_id
        for array_data in problem.extensions.sciml.array_data_files
        for input_id in array_data.inputs
    }


def _array_parameter_layers(problem: core.Problem) -> dict[str, set[str]]:
    """NN parameter arrays across all array data files.

    Returns a dict mapping NN ID to the set of layer IDs that have parameter
    arrays defined.
    """
    result: dict[str, set[str]] = {}
    for array_data in problem.extensions.sciml.array_data_files:
        for nn_id, layers in array_data.parameters.items():
            result.setdefault(nn_id, set()).update(layers)
    return result


def _initial_condition_ids(problem: core.Problem) -> set[str]:
    """Condition IDs applied at the start of any experiment."""
    condition_ids: set[str] = set()
    for experiment in problem.experiments:
        if not experiment.periods:
            continue
        first_period = experiment.sorted_periods[0]
        condition_ids.update(first_period.condition_ids)
    return condition_ids


def _is_array_value(value) -> bool:
    """Whether a target value is the reserved ``array`` keyword."""
    return value is not None and str(value) == ARRAY


# ---------------------------------------------------------------------------
# NN model checks
# ---------------------------------------------------------------------------


class CheckNeuralNetworkModel(lint.ValidationTask):
    """Validate SciML neural network models (YAML format).

    Checks that

    * NN input IDs are valid, unique PEtab identifiers, and
    * all layers and activation functions are supported by the PEtab SciML
      NN model YAML format.
    """

    def run(self, problem: core.Problem) -> lint.ValidationIssue | None:
        messages = []

        for nn in _nn_models(problem):
            nn_id = getattr(nn, "nn_model_id", MISSING_NN_ID)

            # NN input IDs must be valid and unique PEtab identifiers.
            input_ids = [inp.input_id for inp in nn.inputs]
            for input_id in input_ids:
                if not is_valid_identifier(input_id):
                    messages.append(
                        f"Neural network `{nn_id}` has an input ID "
                        f"`{input_id}` that is not a valid PEtab identifier."
                    )
            duplicates = {i for i in input_ids if input_ids.count(i) > 1}
            if duplicates:
                messages.append(
                    f"Neural network `{nn_id}` has duplicate input IDs: "
                    f"{duplicates}."
                )

            # Layers must be supported.
            for layer in nn.layers:
                if layer.layer_type not in Layers:
                    messages.append(
                        f"Neural network `{nn_id}` layer `{layer.layer_id}` "
                        f"uses unsupported layer type `{layer.layer_type}`."
                    )

            # Activation functions (and other functional/method calls in the
            # forward pass) must be supported.
            for node in nn.forward:
                if node.op not in (Op.CALL_FUNCTION, Op.CALL_METHOD):
                    continue
                if node.target in TensorOps:
                    continue
                if node.target not in ActivationFunctions:
                    messages.append(
                        f"Neural network `{nn_id}` uses unsupported "
                        f"activation function/operation `{node.target}`."
                    )

        if messages:
            return lint.ValidationError("\n".join(messages))
        return None


# ---------------------------------------------------------------------------
# Hybridization table checks
# ---------------------------------------------------------------------------


class CheckHybridizationTable(lint.ValidationTask):
    """Validate the SciML hybridization table."""

    def run(self, problem: core.Problem) -> lint.ValidationIssue | None:
        messages = []

        sciml = problem.extensions.sciml
        condition_targets = {
            c.target_id for ct in problem.conditions for c in ct.changes
        }
        nn_inputs, nn_outputs, nn_params = _nn_entity_petab_ids(problem)
        array_input_ids = _array_input_ids(problem)
        array_param_layers = _array_parameter_layers(problem)
        array_param_petab_ids = {
            petab_id
            for petab_id, nn_id in nn_params.items()
            if nn_id in array_param_layers
        }

        hybridizations = sciml.hybridizations
        hyb_target_ids = {hyb.target_id for hyb in hybridizations}

        # Hybridization targets are not also targets in the condition table.
        if culprits := (hyb_target_ids & condition_targets):
            messages.append(
                f"Hybridization target ids `{culprits}` are also "
                "target ids in the condition table."
            )

        # NN inputs must not be used as target values.
        for hyb in hybridizations:
            if hyb.target_value is None:
                continue
            input_culprits = {
                str(s) for s in hyb.target_value.free_symbols
            } & set(nn_inputs)
            if input_culprits:
                messages.append(
                    "The following neural net inputs were used as target "
                    f"values in the hybridization table: `{input_culprits}`."
                )

        # NN outputs must not be target ids (they are values, not targets).
        if culprits := (hyb_target_ids & set(nn_outputs)):
            messages.append(
                f"Neural network outputs `{culprits}` are used as target ids "
                "in the hybridization table, but NN outputs assign values and "
                "thus belong in the target value column."
            )

        for hyb in hybridizations:
            # Target ids must not also be array inputs or NN parameter arrays.
            # Inputs explicitly assigned `array` are exempt: that assignment is
            # realized by the corresponding array data file entry.
            if hyb.target_id in array_input_ids and not _is_array_value(
                hyb.target_value
            ):
                messages.append(
                    f"Hybridization target id `{hyb.target_id}` is also an "
                    "input id in an array data file, but is not assigned "
                    "`array`."
                )
            if hyb.target_id in array_param_petab_ids:
                messages.append(
                    f"Hybridization target id `{hyb.target_id}` refers to "
                    "neural network parameters defined by an array data file "
                    "and cannot be assigned in the hybridization table."
                )

            # Target values must be valid PEtab math expressions. This is
            # normally enforced when the hybridization table is parsed (the
            # value is sympified on construction); this is a defensive
            # re-check in case a target value was assigned some other way.
            try:
                sympify_petab(str(hyb.target_value))
            except Exception as e:
                messages.append(
                    f"Hybridization target value for `{hyb.target_id}` is "
                    f"not a valid PEtab math expression (hint: {e})."
                )

        # Pre-initialization NN inputs must be evaluable before simulation,
        # i.e. only depend on parameters (or `array`), not species or time.
        messages.extend(self._check_pre_initialization(problem, nn_inputs))

        if messages:
            return lint.ValidationError("\n".join(messages))
        return None

    @staticmethod
    def _check_pre_initialization(
        problem: core.Problem, nn_inputs: dict[str, str]
    ) -> list[str]:
        config = _sciml_config(problem)
        if config is None:
            return []

        pre_init_nn_ids = {
            nn_id
            for nn_id, nn_config in (config.neural_networks or {}).items()
            if nn_config.pre_initialization
        }
        if not pre_init_nn_ids:
            return []

        allowed = {p.id for p in problem.parameters} | {ARRAY}
        messages = []
        for hyb in problem.extensions.sciml.hybridizations:
            nn_id = nn_inputs.get(hyb.target_id)
            if nn_id not in pre_init_nn_ids or hyb.target_value is None:
                continue
            invalid = {str(s) for s in hyb.target_value.free_symbols} - allowed
            if invalid:
                messages.append(
                    f"Input `{hyb.target_id}` of pre-initialization neural "
                    f"network `{nn_id}` is assigned a value that depends on "
                    f"`{invalid}`, which is not available before simulation. "
                    "Pre-initialization NN inputs may only depend on "
                    "parameters or array data."
                )
        return messages


# ---------------------------------------------------------------------------
# Condition table checks
# ---------------------------------------------------------------------------


class CheckSciMLConditionTable(lint.ValidationTask):
    """Validate SciML-specific constraints on the condition table."""

    def run(self, problem: core.Problem) -> lint.ValidationIssue | None:
        messages = []

        nn_inputs, nn_outputs, nn_params = _nn_entity_petab_ids(problem)
        array_input_ids = _array_input_ids(problem)
        array_param_layers = _array_parameter_layers(problem)
        array_param_petab_ids = {
            petab_id
            for petab_id, nn_id in nn_params.items()
            if nn_id in array_param_layers
        }

        for condition in problem.conditions:
            for change in condition.changes:
                # Target ids must not also be array inputs or NN parameters.
                if change.target_id in array_input_ids and not _is_array_value(
                    change.target_value
                ):
                    messages.append(
                        f"Condition target id `{change.target_id}` is also an "
                        "input id in an array data file, but is not assigned "
                        "`array`."
                    )
                if change.target_id in array_param_petab_ids:
                    messages.append(
                        f"Condition target id `{change.target_id}` refers to "
                        "neural network parameters defined by an array data "
                        "file and cannot be assigned in the condition table."
                    )

                # NN outputs must not be target ids.
                if change.target_id in nn_outputs:
                    messages.append(
                        f"Neural network output `{change.target_id}` is used "
                        "as a target id in the condition table, but NN "
                        "outputs assign values and belong in the target value "
                        "column."
                    )

                # NN inputs must not be used as target values.
                if change.target_value is not None:
                    input_culprits = {
                        str(s) for s in change.target_value.free_symbols
                    } & set(nn_inputs)
                    if input_culprits:
                        messages.append(
                            "The following neural net inputs were used as "
                            "target values in the condition table: "
                            f"`{input_culprits}`."
                        )

        if messages:
            return lint.ValidationError("\n".join(messages))
        return None


# ---------------------------------------------------------------------------
# Array data file checks
# ---------------------------------------------------------------------------


class CheckArrayDataFiles(lint.ValidationTask):
    """Validate SciML array data files."""

    def run(self, problem: core.Problem) -> lint.ValidationIssue | None:
        messages = []

        initial_conditions = _initial_condition_ids(problem)

        for array_data in problem.extensions.sciml.array_data_files:
            for input_id, input_data in array_data.inputs.items():
                covered: set[str] = set()
                for condition_ids_str in input_data:
                    if condition_ids_str == ALL_CONDITION_IDS:
                        covered = set(initial_conditions)
                        break
                    covered.update(condition_ids_str.split(";"))

                if missing := (initial_conditions - covered):
                    messages.append(
                        f"Array data input `{input_id}` does not define "
                        "arrays for all initial PEtab conditions. Missing: "
                        f"{missing}."
                    )

        if messages:
            return lint.ValidationError("\n".join(messages))
        return None


# ---------------------------------------------------------------------------
# Parameter table checks
# ---------------------------------------------------------------------------


class CheckSciMLParameterTable(lint.ValidationTask):
    """Validate SciML-specific constraints on the parameter table."""

    def run(self, problem: core.Problem) -> lint.ValidationIssue | None:
        messages = []

        array_param_layers = _array_parameter_layers(problem)

        # Map each parameter ID to its NN parameter reference (if any).
        param_nn_ref: dict[str, tuple[str, str | None]] = {}
        for mapping in problem.mappings:
            if not mapping.model_id:
                continue
            if match := NN_PARAMETER_PATTERN.match(mapping.model_id):
                param_nn_ref[mapping.petab_id] = (
                    match.group("nn_id"),
                    match.group("layer"),
                )

        # NN IDs that have parameters (referenced or backed by arrays).
        nns_with_parameters = {
            nn_id for nn_id, _ in param_nn_ref.values()
        } | set(array_param_layers)
        # NN IDs that are declared in the parameter table.
        declared_nn_ids = {
            param_nn_ref[p.id][0]
            for p in problem.parameters
            if p.id in param_nn_ref
        }

        # (1) Any parameter with `nominalValue == array` must be array-backed.
        for parameter in problem.parameters:
            if parameter.nominal_value != ARRAY:
                continue
            if parameter.id not in param_nn_ref:
                messages.append(
                    f"Parameter `{parameter.id}` has nominal value `array` "
                    "but is not mapped to neural network parameters."
                )
                continue
            nn_id, layer = param_nn_ref[parameter.id]
            layers = array_param_layers.get(nn_id)
            if not layers:
                messages.append(
                    f"Parameter `{parameter.id}` has nominal value `array` "
                    f"but no array data is defined for network `{nn_id}`."
                )
            elif layer is not None and layer not in layers:
                messages.append(
                    f"Parameter `{parameter.id}` has nominal value `array` "
                    f"but no array data is defined for layer `{layer}` of "
                    f"network `{nn_id}`."
                )

        # (2) All parameters of each NN must be declared (estimated or fixed).
        for nn_id in sorted(nns_with_parameters - declared_nn_ids):
            messages.append(
                f"Neural network `{nn_id}` has parameters that are not "
                "declared in the parameter table; all NN parameters must be "
                "explicitly set to be estimated or fixed."
            )

        # (3) In "posterior" mode (some parameter has an explicit prior), all
        # estimated NN parameters need a proper prior or finite bounds, so that
        # the implicit uniform(lb, ub) prior is proper.
        posterior = any(
            p.prior_distribution is not None for p in problem.parameters
        )
        if posterior:
            for parameter in problem.parameters:
                if parameter.id not in param_nn_ref or not parameter.estimate:
                    continue
                has_prior = parameter.prior_distribution is not None
                finite_bounds = (
                    parameter.lb is not None
                    and parameter.ub is not None
                    and math.isfinite(parameter.lb)
                    and math.isfinite(parameter.ub)
                )
                if not (has_prior or finite_bounds):
                    messages.append(
                        f"Neural network parameter `{parameter.id}` is "
                        "estimated in a problem with priors, but has "
                        "neither a proper prior nor finite bounds."
                    )

        if messages:
            return lint.ValidationError("\n".join(messages))
        return None
