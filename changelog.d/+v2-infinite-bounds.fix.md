Validate infinite parameter bounds according to PEtab 2.0 erratum E1:
`lowerBound=inf` and `upperBound=-inf` are rejected; if any estimated
parameter has a prior, estimated parameters with infinite bounds
require an explicit prior, which is now checked by `petablint`.
`Problem.get_priors` and `Problem.get_startpoint_distributions` raise a
`ValueError` naming estimated parameters with infinite bounds and no
prior
([#601](https://github.com/PEtab-dev/libpetab-python/pull/601))
