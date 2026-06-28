# Changelog

All notable changes to the `diddesign` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-28

### Added
- Core DID estimator (`did()`) supporting standard DID and staggered adoption (SA) designs
- Generalized K-DID estimation with configurable `kmax` parameter (1-8)
- J-test over-identification test with nested moment deletion (`jtest=True`)
- Sequential DID (sDID), Double-DID, and SA-Double-DID estimators
- Bootstrap inference with cluster-level resampling and deterministic seed management
- Parallel bootstrap via `ThreadPoolExecutor` and `ProcessPoolExecutor` backends
- Diagnostic checks (`did_check()`) with equivalence confidence intervals
- Formula parsing (`did_formula()`) supporting R-style syntax with factor/interaction terms
- Five visualization functions: `plot_estimates`, `plot_trends`, `plot_placebo`, `plot_pattern`, `plot_diagnostics`
- Plotting data preparation via `fit()` and `check()` functions
- Result summary via `summary()` and `format_summary()`
- Structured error code system (E001-E020) aligned with Stata implementation
- Structured warning system (W001-W010) for diagnostic feedback
- Automatic string-to-integer encoding for unit_id and cluster columns
- Verbose parameter (0=quiet, 1=default, 2=progress) for output control
- Immutable result objects using frozen dataclasses and MappingProxyType
- Panel data and repeated cross-section (RCS) support
- Covariate adjustment with numeric, factor, and interaction terms
- Publication-quality matplotlib figures with `style="publication"`

### Dependencies
- numpy >= 1.26
- pandas >= 2.2
- Python >= 3.12
- Optional: matplotlib >= 3.5 (for plotting)
