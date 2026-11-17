For truncated distributions, `Distribution.cdf` now returns 0 below and 1
above the truncation limits instead of values outside [0, 1], and
`Distribution.ppf` returns NaN for quantiles outside [0, 1].
`Distribution.{pdf,cdf,ppf}` now consistently return NumPy scalars for scalar
input (previously sometimes 0-d arrays) and accept lists
([#608](https://github.com/PEtab-dev/libpetab-python/pull/608))
