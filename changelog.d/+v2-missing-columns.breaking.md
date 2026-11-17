v2: Creating a table via `from_df` or `from_tsv` (including the
`Problem.*_df` setters) now raises `petab.v2.MissingColumnsError` if required
columns are missing, naming the table, the missing columns, and the file.
Previously, this led to unspecific errors or, e.g., for missing `estimate` or
`experimentId` columns, was silently accepted. `petablint` reports this error
without a traceback
([#610](https://github.com/PEtab-dev/libpetab-python/pull/610))
