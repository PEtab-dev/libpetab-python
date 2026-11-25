Non-constant numeric expressions in Boolean context (operands of `&&`,
`||`, `!`, `piecewise` conditions) were treated as always `true`.
They are now `true` if non-zero, e.g., `!(x + 1)` is parsed to `x + 1 == 0`
instead of `0.0`
([#583](https://github.com/PEtab-dev/libpetab-python/pull/583))
