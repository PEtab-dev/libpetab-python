`ExperimentsToSbmlConverter` no longer creates an initial assignment with a
branch per experiment for each constant parameter or species that conditions
set to a per-experiment constant. This makes the conversion, and the import
of the converted model, feasible for problems with many experiments
([#579](https://github.com/PEtab-dev/libpetab-python/pull/579))
