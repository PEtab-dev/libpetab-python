Non-constant Booleans in numeric context, e.g., `1 + (x > 0)`, failed
with `TypeError`. They are now converted to `1.0` / `0.0`
([#584](https://github.com/PEtab-dev/libpetab-python/pull/584))
