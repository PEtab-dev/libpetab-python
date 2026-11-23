`BnglModel.get_parameter_value` raises the new `DerivedValueError`
(a subclass of both `ValueError` and `NotImplementedError`) for parameters
defined by an expression.
This fixes a crash of `create_parameter_df` for BNGL models with such
parameters
([#620](https://github.com/PEtab-dev/libpetab-python/pull/620))
