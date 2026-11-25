`petab.v1.distributions.Uniform` requires finite limits with
`low < high`, and `LogUniform` requires `0 < low < high < inf`.
Both raise `ValueError` otherwise. In `petab.v1`, start point sampling
for parameters with infinite bounds, or with `lowerBound=0` and
log-scaled `parameterScaleUniform` priors, now fails when creating the
`Prior` instead of when sampling. Start points can no longer be sampled
for estimated parameters with `lowerBound == upperBound` and no prior
(v1 and v2)
([#604](https://github.com/PEtab-dev/libpetab-python/pull/604))
