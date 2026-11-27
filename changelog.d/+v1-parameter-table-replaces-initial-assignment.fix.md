In the PEtab v1 parameter mapping, a parameter table entry for a model
parameter with a non-numeric initial assignment now replaces that initial
assignment, as for numeric initial assignments and as in PEtab v2.
Previously, such entries were ignored for the model, while condition table
and measurement table references to the same parameter used the parameter
table value. **This changes the mapping, and thus the results, for problems
with such parameter table entries**
([#PR](https://github.com/PEtab-dev/libpetab-python/pull/PR))
