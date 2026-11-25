Fixed `KeyError` in `PySBModel.is_state_variable`, and thus in PEtab v2
validation, for PySB species with binding sites; `(state, bond)` tuples,
`MultiState`, lists of bonds, and complex-level compartments are now
supported
([#592](https://github.com/PEtab-dev/libpetab-python/pull/592))
