PEtab v1 validation: `check_parameter_df`, and thus `lint_problem`, now
reports an error for non-estimated parameters with a missing (empty)
`nominalValue`.
As `petab1to2` validates its input, it now rejects such problems
([#XXX](https://github.com/PEtab-dev/libpetab-python/pull/XXX))
