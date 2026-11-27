PEtab v1 validation: `check_parameter_df`, and thus `lint_problem`, now
reports an error for non-estimated parameters with a missing (empty)
`nominalValue`.
As `petab1to2` validates its input, it now rejects such problems
([#640](https://github.com/PEtab-dev/libpetab-python/pull/640))
