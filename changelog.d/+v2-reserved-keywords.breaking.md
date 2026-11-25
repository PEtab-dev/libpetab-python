PEtab v2: Reserved keywords (`true`, `false`, `inf`, `nan`, `time`, and
PEtab math function names; case-insensitive) are rejected as identifiers in
the PEtab tables, as required by the PEtab v2 specification. Model entities
with such IDs can still be aliased via the mapping table. The keywords are
available as `petab.v2.C.RESERVED_KEYWORDS`
([#609](https://github.com/PEtab-dev/libpetab-python/pull/609))
