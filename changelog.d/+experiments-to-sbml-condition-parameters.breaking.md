`ExperimentsToSbmlConverter` keeps changes that set per-experiment constants
in the condition table, and creates experiment indicators only for
experiments with changes encoded in the model. Consumers of the converted
problem have to apply all changes of the remaining periods, not only the
indicators
([#579](https://github.com/PEtab-dev/libpetab-python/pull/579))
