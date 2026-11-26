`petab1to2`: Fixed a pandas deprecation warning for an all-empty
`observableTransformation` column in the observable table, or an all-empty
`noiseDistribution` column next to an `observableTransformation` column.
With future pandas versions, this would have raised an error
([#635](https://github.com/PEtab-dev/libpetab-python/pull/635))
