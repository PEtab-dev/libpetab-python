`petab.v2.reparametrize_log` and `petab1to2(reparametrize=...)` now replace
`0` overrides of log-transformed placeholders by `-inf` in the measurement
table, as allowed by PEtab 2.0 erratum E3, instead of adding a non-estimated
parameter `negative_infinity`
([#633](https://github.com/PEtab-dev/libpetab-python/pull/633))
