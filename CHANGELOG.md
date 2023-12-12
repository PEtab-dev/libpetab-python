# PEtab changelog

## 0.2 series

### 0.2.5

* Fix accessing `preequilibrationConditionId` without checking for presence
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/228
* Startpoint sampling for a subset of parameters
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/230
* Treat `observableParameter` overrides as placeholders in noise formulae
  by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/231

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.2.4...v0.2.5

### 0.2.4

* Made figure sizes for visualization functions customizable via `petab.visualize.plotting.DEFAULT_FIGSIZE`
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/222
* Fixed Handling missing `nominalValue` in `Problem.get_x_nominal`
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/223
* Fixed pandas 2.1.0 `FutureWarnings`
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/226
* Added pre-commit-config, ran black, isort, ...
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/225

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.2.3...v0.2.4

### 0.2.3

* Fixed validation failures in case of missing optional fields in visualization tables
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/214
* Make validate_visualization_df work without matplotlib installation
  by @dweindl @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/215

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.2.2...v0.2.3

### 0.2.2

* Fixed IndexError with numpy 1.25.0 by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/209
* Made `SbmlModel.from_file(..., model_id)` optional by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/207

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.2.1...v0.2.2

### 0.2.1

Fixes:
* Fixed an issue in `Problem.to_files(model_file=...)` (#204)
* Fixed `PySBModel.get_parameter_value`, which incorrectly returned the parameter name instead of its value (#203)

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.2.0...v0.2.1

### 0.2.0

Note: petab 0.2.0 requires Python>=3.9

Features:
* Plot measurements for t = 'inf'
  by @plakrisenko in https://github.com/PEtab-dev/libpetab-python/pull/149
* Added validation for visualization files
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/184
  https://github.com/PEtab-dev/libpetab-python/pull/189
* Startpoints as dict
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/188
* Residuals plot
  by @plakrisenko in https://github.com/PEtab-dev/libpetab-python/pull/187
  https://github.com/PEtab-dev/libpetab-python/pull/191
* add goodness of fit plot
  by @plakrisenko in https://github.com/PEtab-dev/libpetab-python/pull/192
* Add PySBModel for handling of PySB models
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/145

Fixes
* Vis: Don't fail on missing simulations
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/185
* prevent strings being parsed as nan in get_visualization_df
  by @plakrisenko in https://github.com/PEtab-dev/libpetab-python/pull/193
* Fix get_model_for_condition
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/194
* Simulator: rename measurement column to simulation
  by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/199
* Fix sympy symbol name clashes
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/202

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.1.30...v0.2.0

## 0.1 series

### 0.1.30

Various smaller fixes:

* Vis: Handle missing data more gracefully by @dweindl
  in https://github.com/PEtab-dev/libpetab-python/pull/175
* Fix test dependencies: scipy by @dweindl
  in https://github.com/PEtab-dev/libpetab-python/pull/177
* Add `petab.Problem.__str__` by @dweindl
  in https://github.com/PEtab-dev/libpetab-python/pull/178
* Fix deprecated tight layout matplotlib by @yannikschaelte
  in https://github.com/PEtab-dev/libpetab-python/pull/180
* Move tests to tox by @yannikschaelte
  in https://github.com/PEtab-dev/libpetab-python/pull/182
* Update deprecated functions in tests by @yannikschaelte
  in https://github.com/PEtab-dev/libpetab-python/pull/181
* Use petab identifier for combine archives by @fbergmann
  in https://github.com/PEtab-dev/libpetab-python/pull/179

New Contributors
* @fbergmann made their first contribution
  in https://github.com/PEtab-dev/libpetab-python/pull/179

**Full Changelog**:
https://github.com/PEtab-dev/libpetab-python/compare/v0.1.29...v0.1.30

### 0.1.29

Features:
* Method to unflatten simulation dataframe produced by flattened PEtab problem
  by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/171
* Methods to simplify PEtab problems
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/172

Fixes:
* Fix relative paths for model files
  by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/173

**Full Changelog**
https://github.com/PEtab-dev/libpetab-python/compare/v0.1.28...v0.1.29

### 0.1.28

* Fixed validation for output parameters columns in the condition table
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/161
* Added Python support policy
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/162
* Fixed typehints and deprecation warning
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/165
* Fixed SBML validation
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/168
* Fixed deprecation warning from `get_model_for_condition`
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/169

**Full Changelog**:
https://github.com/PEtab-dev/libpetab-python/compare/v0.1.27...v0.1.28

### 0.1.27

Features:
* Added method to check if measurement time is at steady-state by @dilpath in
  https://github.com/PEtab-dev/libpetab-python/pull/124
* Create dummy simulation conditions dataframe for empty measurements by
  @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/127
* Validator: Report empty noiseFormula by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/134
* Speedup visspec assembly / fix deprecation warning by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/135
* Handle incomplete PEtab problems in `petab.Problem.from_yaml` by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/138
* Argument forwarding for
  `Problem.get_optimization_to_simulation_parameter_mapping` by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/159
* Added candidate schema for version 2  by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/142
* `get_parameter_df`: Allow any collection of parameter tables by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/153,
  @m-philipps in https://github.com/PEtab-dev/libpetab-python/pull/156,
  @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/157
* Updated visualization example notebooks
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/137,
  by @plakrisenko in https://github.com/PEtab-dev/libpetab-python/pull/146,
  by @plakrisenko in https://github.com/PEtab-dev/libpetab-python/pull/147
* Added support for PEtab problems with multiple condition files
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/152
* Added abstraction for (SBML) models by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/133

Fixes:
* Apply get table method before write table method to ensure correct index
  by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/126
* petablint: Fix incorrect noise-parameter-mismatch error message
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/129
* Fixed handling of NaN values for parameters in condition table
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/150
* More informative `petab.calculate` errors
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/151

Removals:
* Removed ancient/deprecated default file naming scheme
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/132
* Removed ancient deprecated functions related to specifying observables/noise
  models inside SBML
  by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/140
  https://github.com/PEtab-dev/libpetab-python/pull/131
* Removed deprecated visualization functions
   by @dweindl in https://github.com/PEtab-dev/libpetab-python/pull/130

**New Contributors**
* @m-philipps made their first contribution in https://github.com/PEtab-dev/libpetab-python/pull/156

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.1.26...v0.1.27

### 0.1.26

* Fix SBML Rule handling logic by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/120

### 0.1.25
* Fix for pytest 7.1 by @yannikschaelte in https://github.com/PEtab-dev/libpetab-python/pull/112
* Fix jinja version by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/115
* Add steady state constant by @dilpath in https://github.com/PEtab-dev/libpetab-python/pull/114
* Omit measurement processing if not relevant for parameter mapping by @FFroehlich in https://github.com/PEtab-dev/libpetab-python/pull/117

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.1.24...v0.1.25

### 0.1.24

* Added method to generate condition-specific SBML models by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/108
* GHA: Regular package installation instead of -e by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/106
* Fixed unclosed file warnings by @dilpath in
  https://github.com/PEtab-dev/libpetab-python/pull/107

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.1.23...v0.1.24

### 0.1.23

* Added command line interface for plotting by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/98
* Fixed petab.visualize.data_overview.create_report by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/96,
  https://github.com/PEtab-dev/libpetab-python/pull/104
* Vis: Fixed cropped errorbars by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/99
* Fixed pandas deprecation warning by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/103

... and other changes by @plakrisenko, @dweindl

**Full Changelog**: https://github.com/PEtab-dev/libpetab-python/compare/v0.1.22...v0.1.23

### 0.1.22

* Allow zero bounds for log parameters by @FFroehlich in
  https://github.com/PEtab-dev/libpetab-python/pull/83
* Adapt to Matplotlib 3.5 by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/86
* Allow specifying file format for visualization by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/85
* Visualization: Don't mess with rcParams by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/90
* Linter: Check condition IDs are unique by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/92
* Add support for `pathlib` for reading PEtab tables by @dweindl, @dilpath
  in https://github.com/PEtab-dev/libpetab-python/pull/93,
  https://github.com/PEtab-dev/libpetab-python/pull/91
* Run tests also on Python 3.10 by @dweindl in
  https://github.com/PEtab-dev/libpetab-python/pull/88
* Fix remote file retrieval on Windows @dweindl, @dilpath
  in https://github.com/PEtab-dev/libpetab-python/pull/91
* Fix test suite for Windows @dweindl, @dilpath
  in https://github.com/PEtab-dev/libpetab-python/pull/91

**Full Changelog**:
https://github.com/PEtab-dev/libpetab-python/compare/v0.1.21...v0.1.22

### 0.1.21

* PEtab spec compliance: measurements must now be not null, and numeric (#76)
  * Users who relied on null measurements for simulation/plotting are
    recommended to store these dummy simulation-only measurements in an
    additional file, separate to real measurements used for calibration
* Improve Unicode support (#79, fixes #77)
* Convenience methods to scale or unscale a parameter vector (#78)

### 0.1.20

* Visualization: plot additional simulation points (not only at measurements) (#62), bugfix (#68)
* Documentation: visualization, observables, simulation, Sphinx fixes (#67)
* Lint: ensure valid parameter IDs in observable and noise parameters (#69)
* Convenience method for quick export of a PEtab problem to files (#71)

### 0.1.19

* Visualization: refactoring (#58) including various bug fixes
* Validation: Fixed detection of missing observable/noise parameter overrides
  (#64)
* Optional relative paths in generated YAML (#57)

### 0.1.18

* Fixed various documentation issues
* Parameter mapping: Added option to ignore time-point specific
  noiseParameters (#51)

### 0.1.17

* Updated package URL
* Fixed noise formula check (#49)
* Fixed override check and add noise formula check (#48)
* Fixed timepoint override check (#47)

### 0.1.16

Update python version for pypi deployment, no further changes

### 0.1.15

**NOTE:** The original PEtab format + petab package repository has been split
up (PEtab-dev/libpetab-python#41). This repository now only contains the petab
Python package. The PEtab specifications and related information are available
at https://github.com/PEtab-dev/PEtab.

* Improved `petab.flatten_timepoint_specific_output_overrides` (PEtab-dev/libpetab-python#42)
* Validator: output message in case of successful check is added (PEtab-dev/PEtab#487)
* Update how-to-cite (Closes PEtab-dev/PEtab#432) (PEtab-dev/PEtab#509)
* Broadcast and mapping of scale and unscale functions (PEtab-dev/PEtab#505)
* Update Python requirement (3.7.1) (PEtab-dev/PEtab#502)
* Fix `petab.get_required_parameters_for_parameter_table`
  (PEtab-dev/libpetab-python#43)
* Fix `petab.measurement_table_has_timepoint_specific_mappings`
  (PEtab-dev/libpetab-python#44)

### 0.1.14

* Fix sampling of priors in `parameterScale` (PEtab-dev/PEtab#492)
* Clarify documentation of `parameterScale` priors
* Improvements in `petab.simulate` (PEtab-dev/PEtab#479):
  * Fix default noise distributions
  * Add option for non-negative synthetic data

### 0.1.13

* Fix for pandas 1.2.0 -- use `get_handle` instead of `get_filepath_or_buffer`
* Fix erroneous `petab_test_suite` symlink (all PEtab-dev/PEtab#493)

### 0.1.12

* Documentation update:
  * Added SBML2Julia to list of tools supporting PEtab
  * Extended PEtab introduction
  * Tutorial for creating PEtab files
* Minor fix: Default argument for optional 'model' parameter in
  `petab.lint.check_condition_df`` (PEtab-dev/PEtab#477)

### 0.1.11

* Function for generating synthetic data (PEtab-dev/PEtab#472)
* Minor documentation updates (PEtab-dev/PEtab#470)

### 0.1.10

* Fixed deployment setup, no further changes.*

### 0.1.9

Library:

* Allow URL as filenames for YAML files and SBML models (Closes PEtab-dev/PEtab#187) (PEtab-dev/PEtab#459)
* Allow model time in observable formulas (PEtab-dev/PEtab#445)
* Make float parsing from CSV round-trip (PEtab-dev/PEtab#444)
* Validator: Error message for missing IDs, with line numbers. (PEtab-dev/PEtab#467)
* Validator: Detect duplicated observable IDs (PEtab-dev/PEtab#446)
* Some documentation and CI fixes / updates
* Visualization: Add option to save visualization specification (PEtab-dev/PEtab#457)
* Visualization: Column XValue not mandatory anymore (PEtab-dev/PEtab#429)
* Visualization: Add sorting of indices of dataframes for the correct sorting
  of x-values (PEtab-dev/PEtab#430)
* Visualization: Default value for the column x_label in vis_spec (PEtab-dev/PEtab#431)


### 0.1.8

Library:

* Use ``core.is_empty`` to check for empty values (PEtab-dev/PEtab#434)
* Move tests to python 3.8 (PEtab-dev/PEtab#435)
* Update to libcombine 0.2.6 (PEtab-dev/PEtab#437)
* Make float parsing from CSV round-trip (PEtab-dev/PEtab#444)
* Lint: Allow model time in observable formulas (PEtab-dev/PEtab#445)
* Lint: Detect duplicated observable ids (PEtab-dev/PEtab#446)
* Fix likelihood calculation with missing values (PEtab-dev/PEtab#451)

Documentation:

* Move format documentation to restructuredtext format (PEtab-dev/PEtab#452)
* Document all noise distributions and observable scales (PEtab-dev/PEtab#452)
* Fix documentation for prior distribution (PEtab-dev/PEtab#449)

Visualization:

* Make XValue column non-mandatory (PEtab-dev/PEtab#429)
* Apply correct condition sorting (PEtab-dev/PEtab#430)
* Apply correct default x label (PEtab-dev/PEtab#431)


### 0.1.7

Documentation:

* Update coverage and links of supporting tools
* Update explanatory figure


### 0.1.6

Library:

* Fix handling of empty columns for residual calculation (PEtab-dev/PEtab#392)
* Allow optional fixing of fixed parameters in parameter mapping (PEtab-dev/PEtab#399)
* Fix function to flatten out time-point specific overrides (PEtab-dev/PEtab#404)
* Add function to create a problem yaml file (PEtab-dev/PEtab#398)
* Allow merging of multiple parameter files (PEtab-dev/PEtab#407)

Documentation:

* In README, add to the overview table the coverage for the supporting tools,
  and links and usage examples (various commits)
* Show REAMDE on readthedocs documentation front page (PEtab-dev/PEtab#400)
* Correct description of observable and noise formulas (PEtab-dev/PEtab#401)
* Update documentation on optional visualization values (PEtab-dev/PEtab#405, PEtab-dev/PEtab#419)

Visualization:

* Fix sorting problem (PEtab-dev/PEtab#396)
* More generously handle optional values (PEtab-dev/PEtab#405, PEtab-dev/PEtab#419)
* Create dataset id also for simulation dataframe (PEtab-dev/PEtab#408)
* Extend test suite for visualization (PEtab-dev/PEtab#418)


### 0.1.5

Library:

* New create empty observable function (issue 386) (PEtab-dev/PEtab#387)
* Deprecate petab.sbml.globalize_parameters (PEtab-dev/PEtab#381)
* Fix computing log10 likelihood (PEtab-dev/PEtab#380)
* Documentation update and typehints for visualization  (PEtab-dev/PEtab#372)
* Ordered result of `petab.get_output_parameters`
* Fix missing argument to parameters.create_parameter_df

Documentation:
* Add overview of supported PEtab feature in toolboxes
* Add contribution guide
* Fix optional values in documentation (PEtab-dev/PEtab#378)


### 0.1.4

Library:

* Fixes / updates in functions for computing llh and chi2
* Allow and require output parameters defined in observable table to be defined in parameter table
* Fix merge_preeq_and_sim_pars_condition which incorrectly assumed lists
  instead of dicts
* Update parameter mapping to deal with species and compartments in
  condition table
* Removed `petab.migrations.sbml_observables_to_table`

  For converting older PEtab files to observable table format, use one of the
  previous releases

* Visualization:
  * Fix various issues with get_data_to_plot
  * Fixed various issues with expected presence of optional columns


### 0.1.3

File format:

* Updated documentation
* Observables table in YAML file now mandatory in schema (was implicitly
  mandatory before, as observable table was required already)

Library:
* petablint:
  * Fix: allow specifying observables file via CLI (Closes PEtab-dev/PEtab#302)
  * Fix: nominalValue is optional unless estimated!=1 anywhere (Fixes PEtab-dev/PEtab#303)
  * Fix: handle undefined observables more gracefully (Closes PEtab-dev/PEtab#300) (PEtab-dev/PEtab#351)
* Parameter mapping:
  * Fix / refactor parameter mapping (breaking change) (PEtab-dev/PEtab#344)
    (now performing parameter value and scale mapping together)
  * check optional measurement cols in mapping (PEtab-dev/PEtab#350)
* allow calculating llhs (PEtab-dev/PEtab#349), chi2 values (PEtab-dev/PEtab#348) and residuals (PEtab-dev/PEtab#345)
* Visualization
  * Basic Scatterplots & lot of bar plot fixes (PEtab-dev/PEtab#270)
  * Fix incorrect length of bool `bool_preequ` when subsetting with ind_meas
    (Closes PEtab-dev/PEtab#322)
* make libcombine optional (PEtab-dev/PEtab#338)


### 0.1.2

Library:

* Extensions and fixes for the visualization functions (PEtab-dev/PEtab#255, PEtab-dev/PEtab#262)
* Allow to extract fixed|free and scaled|non-scaled parameters (PEtab-dev/PEtab#256, PEtab-dev/PEtab#268, PEtab-dev/PEtab#273)
* Various fixes (esp. PEtab-dev/PEtab#264)
* Add function to get observable ids (PEtab-dev/PEtab#269)
* Improve documentation (esp. PEtab-dev/PEtab#289)
* Set default column for simulation results to 'simulation'
* Add support for COMBINE archives (PEtab-dev/PEtab#271)
* Fix sbml observables to table
* Improve prior and dataframe tests (PEtab-dev/PEtab#285, PEtab-dev/PEtab#286, PEtab-dev/PEtab#297)
* Add function to get parameter table with all default values (PEtab-dev/PEtab#288)
* Move tests to github actions (PEtab-dev/PEtab#281)
* Check for valid identifiers
* Fix handling of empty values in dataframes
* Allow to get numeric values in parameter mappings in scaled form (PEtab-dev/PEtab#308)

### 0.1.1

Library:

* Fix parameter mapping: include output parameters not present in SBML model
* Fix missing `petab/petab_schema.yaml` in source distribution
* Let get_placeholders return an (ordered) list of placeholders
* Deprecate `petab.problem.from_folder` and related functions
  (obsolete after introducing more flexible YAML files for grouping tables
  and models)

### 0.1.0

Data format:

* Introduce observables table instead of SBML assignment rules for defining
  observation model (PEtab-dev/PEtab#244) (moves observableTransformation and noiseModel
  from the measurement table to the observables table)
* Allow initial concentrations / sizes in condition table (PEtab-dev/PEtab#238)
* Fixes and clarifications in the format documentation
* Changes in prior columns of the parameter table (PEtab-dev/PEtab#222)
* Introduced separate version number of file format, this release being
  version 1

Library:

* Adaptations to new file formats
* Various bugfixes and clean-up, especially in visualization and validator
* Parameter mapping changed to include all model parameters and not only
  those differing from the ones defined inside the SBML model
* Introduced constants for all field names and string options, replacing
  most string literals in the code (PEtab-dev/PEtab#228)
* Added unit tests and additional format validation steps
* Optional parallelization of parameter mapping (PEtab-dev/PEtab#205)
* Extended documentation (in-source and example Jupyter notebooks)

### 0.0.2

Bugfix release

* Fix `petablint` error
* Fix minor issues in `petab.visualize`

### 0.0.1

Data format:
* Update format and documentation with respect to data and parameter scales
  (PEtab-dev/PEtab#169)
* Define YAML schema for grouping PEtab files, also allowing for more complex
  combinations of files (PEtab-dev/PEtab#183)

Library:
* Refactor library. Reorganize `petab.core` functions.
* Fix visualization w/o condition names PEtab-dev/PEtab#142
* Extend validator
* Removed deprecated functions petab.Problem.get_constant_parameters
  and petab.sbml.constant_species_to_parameters
* Minor fixes and extensions

## 0.0 series

### 0.0.0a17

Data format: *No changes*

Library:
* Extended visualization support
* Add helper function and test case to deal with timepoint-specific parameters
  flatten_timepoint_specific_output_overrides (PEtab-dev/PEtab#128) (Closes PEtab-dev/PEtab#125)
* Fix get_noise_distributions: so far we got 'normal' everywhere due to
  wrong grouping (PEtab-dev/PEtab#147)
* Fix create_parameter_df: Exclude rule targets (PEtab-dev/PEtab#149)
* Verify condition table column names occur as model parameters
  (Closes PEtab-dev/PEtab#150) (PEtab-dev/PEtab#151)
* More informative error messages in case of wrongly set observable and
  noise parameters (Closes PEtab-dev/PEtab#118) (PEtab-dev/PEtab#155)
* Update doc for copasi import and github installation (PEtab-dev/PEtab#158)
* Extend validator to check if all required parameters are present in
  parameter table (Closes PEtab-dev/PEtab#43) (PEtab-dev/PEtab#159)
* Setup documentation for RTD (PEtab-dev/PEtab#161)
* Handle None in petab.core.split_parameter_replacement_list (Closes PEtab-dev/PEtab#121)
* Fix(lint) correct handling of optional columns. Check before access.
* Remove obsolete generate_experiment_id.py (Closes PEtab-dev/PEtab#111) PEtab-dev/PEtab#166

### 0.0.0a16 and earlier

See git history
