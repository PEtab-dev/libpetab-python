from __future__ import annotations

from .. import core, lint

__all__ = [
    "CheckHybridizationTable",
]


class CheckHybridizationTable(lint.ValidationTask):
    """Validate the SciML hybridization table."""

    def run(self, problem: core.Problem) -> lint.ValidationIssue | None:
        messages = []

        condition_targets = {
            c.target_id for ct in problem.conditions for c in ct.changes
        }
        # Only YAML-format networks are loaded as NNModel objects
        nn_input_ids = {
            inp.input_id
            for nn in problem.extensions.sciml.neural_networks
            if hasattr(nn, "inputs")
            for inp in nn.inputs
        }
        hyb_target_ids = {
            hyb.target_id for hyb in problem.extensions.sciml.hybridizations
        }
        hyb_target_vals = {
            hyb.target_value for hyb in problem.extensions.sciml.hybridizations
        }

        # Hybridization targets are not also targets in the condition table
        if culprits := (hyb_target_ids & condition_targets):
            messages.append(
                f"Hybridization target ids `{culprits}` are also "
                "target ids in the condition table."
            )
        # NN inputs are not used as target values
        if culprits := (hyb_target_vals & nn_input_ids):
            messages.append(
                "The following neural net inputs were used as target values "
                f"in the Hybridization table: `{culprits}`."
            )

        if messages:
            return lint.ValidationError("\n".join(messages))

        return None
