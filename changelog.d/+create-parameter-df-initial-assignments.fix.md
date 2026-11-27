`petab.v1.create_parameter_df(..., include_optional=True)` no longer includes
parameters whose initial assignment is not closed-form (e.g., refers to
model entities, function definitions, or time), since listing them in the
parameter table would replace the initial assignment. Such parameters are
still included if they are referenced in the measurement table or the
condition table, since these references require a parameter table entry;
in this case, the parameter table entry also replaces the initial
assignment. For parameters with closed-form initial assignments
(e.g., `2 * 3`), `create_parameter_df` now uses the value of the assignment
as nominal value, instead of the SBML `value` attribute
([#PR](https://github.com/PEtab-dev/libpetab-python/pull/PR))
