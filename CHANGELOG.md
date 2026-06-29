# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-29

### Fixed

- Corrected project URLs in package metadata (Homepage, Documentation, Issues, Source).

## [0.1.0] - 2026-06-29

### Added

- Core estimators: DID, Double-DID, K-DID, SA-DID, SA-Double-DID, SA-K-DID
- J-test overidentification moment selection for K-DID (Hansen 1982)
- Pre-treatment diagnostics: placebo tests, equivalence confidence intervals, trend comparison
- Five visualization functions: plot_estimates, plot_trends, plot_placebo, plot_diagnostics, plot_pattern
- Publication-quality plotting with "publication" style preset
- Formula interface supporting covariates, factor variables, and interactions
- Structured error system with guided messages (E001-E020, W001-W012)
- Built-in datasets: Malesky et al. (2014), Paglayan (2019)
- Immutable result objects with pandas DataFrame accessors
- LaTeX table export via to_latex()
- Full type annotations with py.typed marker
- Sphinx documentation for ReadTheDocs deployment
