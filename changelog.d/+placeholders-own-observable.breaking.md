PEtab v1 validation: `lint_problem` now reports an error for placeholder
parameters that are model entities, condition table columns, mapping table
entries or observable IDs, or that are used in another observable, the
condition table or the measurement table, and for noise placeholders used in
the observable formula
(new `assert_placeholders_only_used_in_own_observable`).
As `petab1to2` validates its input, it now rejects such problems
([#605](https://github.com/PEtab-dev/libpetab-python/pull/605))
