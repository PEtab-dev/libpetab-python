For truncated distributions, `Distribution.cdf` now returns 0 below and 1
above the truncation limits instead of values outside [0, 1], and
`Distribution.ppf` returns NaN for quantiles outside [0, 1]. `pdf`, `cdf` and
`ppf` of truncated distributions now also accept lists
([#608](https://github.com/PEtab-dev/libpetab-python/pull/608))
