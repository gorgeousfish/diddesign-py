# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-06-29

### Fixed

- `report()` no longer crashes on RCS design results.
- J-test statistics now displayed in `print(result)` and `format_summary()` when `jtest=True`.
- Cleaned `__all__` to hide internal implementation names from public API.
- Parameter validation errors now use structured `[E0xx]` format with suggestions.

### Added

- `verbose` parameter in `did()` for bootstrap progress reporting.

## [0.1.5] - 2026-06-29

### Fixed

- Plot functions no longer call `plt.show()` unconditionally (returns Figure for user control).
- E002 error suggestion now correctly describes formula vs. parameter conflict.
- Copyright year updated to 2026.

### Added

- Data loaders (`load_malesky2014`, `load_paglayan2019`) now accessible from top-level `diddesign` namespace.

## [0.1.4] - 2026-06-29

### Changed

- Installation instructions updated to use PyPI (`pip install diddesign`).

## [0.1.3] - 2026-06-29

### Fixed

- Fixed PyPI author display to show both authors (Xuanyu Cai, Wenli Xu).

## [0.1.2] - 2026-06-29

### Changed

- Development status upgraded to Production/Stable.
- Added Wenli Xu to maintainers for full PyPI visibility.

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
