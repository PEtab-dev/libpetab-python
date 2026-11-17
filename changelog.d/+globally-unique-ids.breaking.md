PEtab v2 validation: new validation task `CheckGloballyUniqueIds`:
identifiers, including observable and noise placeholders, must be unique
within the problem and must not be model entity IDs. It replaces
`CheckObservablesDoNotShadowModelEntities` in the default validation tasks
and the corresponding checks in `CheckMappingTable`
([#606](https://github.com/PEtab-dev/libpetab-python/pull/606))
