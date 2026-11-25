`sympify_petab` now returns top-level Boolean expressions as numeric
expressions, e.g., `x > 0` as `Piecewise((1.0, x > 0), (0.0, True))`
instead of a relational
([#584](https://github.com/PEtab-dev/libpetab-python/pull/584))
