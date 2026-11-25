PEtab v2 validation: the new validation task `CheckModelEntityEligibility`
reports errors for model entities that the PEtab v2 specification does not
allow as condition targets (e.g., targets of assignment rules, reactions) or
in the parameter table (e.g., targets of assignment rules, rate rules, or
event assignments), and warnings for questionable but valid uses.
Currently implemented for SBML models
([#623](https://github.com/PEtab-dev/libpetab-python/pull/623))
