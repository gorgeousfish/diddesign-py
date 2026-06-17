# diddesign: Double Difference-in-Differences for Python

[![Python >=3.12](https://img.shields.io/badge/python-%E2%89%A53.12-blue)](https://www.python.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green)](LICENSE)

> Python implementation of the multiple-pre-treatment Difference-in-Differences
> estimator proposed by Egami & Yamauchi (2023, *Political Analysis*).

## Overview

`diddesign` implements the multiple-pre-treatment DID framework that combines
standard DID and sequential DID estimators via an efficient GMM weighting
scheme. The **Double DID** estimator exploits all available pre-treatment
periods to produce a more efficient combined estimate under weaker
identification assumptions than either component alone.

**Three key advantages of the Double DID approach:**

1. **Weaker assumptions** — Requires only one of the two standard parallel-trends
   conditions (either the conventional or the sequential version) to hold,
   providing partial robustness to violations.
2. **Efficiency gains** — The GMM-optimal combination of DID and sequential DID
   achieves lower variance than either component when both identification
   conditions hold.
3. **Built-in diagnostics** — Pre-treatment placebo tests, trend visualizations,
   and pattern checks are accessible from a single function call.

**Package highlights:**

- Native Python/pandas workflow — fit, inspect, export, and plot without
  leaving the Python ecosystem.
- Immutable `DidResult` objects with frame accessors for estimates, bootstrap
  draws, GMM weights, and covariance-matrix rows.
- K-DID extension for panels with more than two pre-treatment periods
  (`kmax ≥ 3`).
- J-test overidentification check with adaptive moment selection (`jtest=True`).
- Staggered-adoption design with lead-specific estimates (`design="sa"`).
- Publication-grade matplotlib visualizations built in.
- Parallel bootstrap via `parallel=True`.
- Covariate adjustment with `factor()` categorical encoding and `x1:x2`
  interaction terms.

## Key Concepts

### Identification Assumptions

| Assumption | Formal Name | Applies To |
| --- | --- | --- |
| A1 | Parallel Trends (PT) | Standard DID |
| A2 | Parallel Trends in Trends (PTT) | Sequential DID |
| A1 ∪ A2 | At least one holds | Double DID |

### Estimator–Assumption Mapping

| Estimator | Required Assumptions | Available Periods |
| --- | --- | --- |
| DID | A1 (Parallel Trends) | ≥ 1 pre, 1 post |
| Sequential DID (sDID) | A2 (Parallel Trends in Trends) | ≥ 2 pre, 1 post |
| Double DID | A1 ∪ A2 | ≥ 2 pre, 1 post |
| K-DID | A1 ∪ A2 ∪ higher-order | ≥ K pre, 1 post |

### When To Use Each Estimator

| Scenario | Recommendation |
| --- | --- |
| Standard 2-period panel | Use DID alone (`kmax=1`) |
| ≥ 2 pre-treatment periods, want robustness | Use Double DID (default `kmax=2`) |
| ≥ 3 pre-treatment periods, want efficiency | Use K-DID (`kmax=3` or higher) |
| Multiple treatment timing groups | Use staggered adoption (`design="sa"`) |
| Suspect non-parallel pre-trends | Run `did_check()` first |

## Installation

Install from a local checkout (editable mode):

```bash
python3 -m pip install -e diddesign-py
```

Or install the built wheel:

```bash
pip install diddesign-0.1.0-py3-none-any.whl
```

For visualization support (matplotlib):

```bash
pip install "diddesign[plot]"
```

**Requirements:** Python ≥ 3.12, NumPy ≥ 1.26, pandas ≥ 2.2.

## When To Use `diddesign`

Use `diddesign` when the analysis targets the Egami-Yamauchi
multiple-pre-treatment DID design and the reporting task needs access to the
component estimates, GMM weights, diagnostics, bootstrap draws, and plotting rows
after fitting.

For adjacent DID tasks, other Python packages may be the better first choice:
PyFixest for fixed-effects DID and event-study regression workflows,
`differences` for cohort-time ATT(g,t) estimation, DoubleML for DID with
machine-learning nuisance estimation, CausalPy for broader quasi-experimental
Bayesian/OLS workflows, ModernDiD for broad modern-DID estimator coverage,
`lwdid` for Lee-Wooldridge rolling-transformation DID, `sdid` for synthetic
Difference-in-Differences, and `diff-diff` for a broader scikit-learn-style DID
toolkit. `diddesign` is narrower: it focuses on returning inspectable Python
objects for multiple-pre-treatment Double DID and staggered adoption
calculations.

## Quick Start

### Basic DID (Panel Data)

```python
import numpy as np
import pandas as pd
from diddesign import did, summary

# Simulate a balanced panel: 100 units, 3 time periods
np.random.seed(42)
n_units = 100
times = [2019, 2020, 2021]
units = np.repeat(range(n_units), len(times))
time_col = np.tile(times, n_units)
treated = units < n_units // 2
post = time_col == 2021
treatment_effect = 2.0

y = np.random.normal(0, 1, n_units * len(times))
y += treatment_effect * (treated & post)

data = pd.DataFrame({
    "unit": units,
    "time": time_col,
    "treat": (treated & post).astype(int),
    "y": y,
})

# Fit Double DID
result = did(data, formula="y ~ treat", time="time", unit_id="unit",
             n_boot=50, random_seed=42)

# Print formatted summary
print(summary(result, as_frame=True))
```

### Basic DID (Repeated Cross-Section)

```python
import numpy as np
import pandas as pd
from diddesign import did

np.random.seed(123)
n_per_period = 200
# Three time periods needed for Double DID: 2 pre-treatment + 1 post
rows = []
for t in [0, 1, 2]:
    for i in range(n_per_period):
        treat_group = 1 if i < n_per_period // 2 else 0
        post_indicator = 1 if t == 2 else 0
        y = 5.0 + 2.0 * treat_group + 0.5 * t + 3.0 * treat_group * post_indicator
        y += np.random.normal(0, 1)
        rows.append({"y": y, "treat": treat_group, "time": t, "post": post_indicator})

data = pd.DataFrame(rows)

result = did(data, outcome="y", treatment="treat", time="time",
             post="post", data_type="rcs", n_boot=50, random_seed=123)

# Access estimates as a DataFrame
print(result.to_estimates_frame())
```

### Staggered Adoption Design

```python
import numpy as np
import pandas as pd
from diddesign import did, fit

np.random.seed(99)
states = list(range(20))
years = list(range(2000, 2010))
rows = []
# Assign staggered treatment: first 10 states treated at year 2005
for s in states:
    treat_year = 2005 if s < 10 else None
    for t in years:
        treated = 1 if (treat_year and t >= treat_year) else 0
        y = 10 + 0.5 * t + 2.0 * treated + np.random.normal(0, 1)
        rows.append({"state": s, "year": t, "treat": treated, "y": y})

data = pd.DataFrame(rows)

result = did(data, outcome="y", treatment="treat", time="year",
             unit_id="state", design="sa", lead=(0, 1, 2),
             thres=1, n_boot=50, random_seed=99)

# Lead-specific estimates
print(result.to_estimates_frame())
# Fit rows for event-study plot
print(fit(result, as_frame=True))
```

### K-DID with Multiple Pre-treatment Periods

```python
import numpy as np
import pandas as pd
from diddesign import did

np.random.seed(7)
n_units = 80
times = [2016, 2017, 2018, 2019, 2020]  # 4 pre-treatment periods
units = np.repeat(range(n_units), len(times))
time_col = np.tile(times, n_units)
treated = units < n_units // 2
post = time_col == 2020

y = np.random.normal(0, 1, n_units * len(times))
y += 1.5 * (treated & post)

data = pd.DataFrame({
    "unit": units, "time": time_col,
    "treat": (treated & post).astype(int), "y": y,
})

# Request K-DID with kmax=3 and J-test moment selection
result = did(data, formula="y ~ treat", time="time", unit_id="unit",
             kmax=3, jtest=True, n_boot=100, random_seed=7)

# K-DID combined estimate alongside component rows
print(result.to_estimates_frame())
# Bootstrap draws with component columns
print(result.to_bootstrap_frame().head())
```

### K-DID with Staggered Adoption

When multiple pre-treatment periods are available in a staggered-adoption
design, use `kmax > 2` to employ higher-order differencing with GMM-optimal
combination and over-identification testing:

```python
import numpy as np
import pandas as pd
from diddesign import did

# Load staggered-adoption panel data
data = pd.read_stata("paglayan2019.dta")
data["log_expenditure"] = np.log(data["pupil_expenditure"] + 1.0)

result = did(
    data,
    outcome="log_expenditure",
    treatment="treatment",
    time="year",
    unit_id="state",
    design="sa",
    kmax=3,          # Use up to 3rd-order differencing
    jtest=True,      # Enable J-test for over-identification
    lead=(0, 1, 2),
    thres=1,
    n_boot=100,
    random_seed=2024,
)

# Component estimates (SA-DID, SA-sDID, SA-kDID-3)
# and GMM-combined estimate (SA-K-DID) are in:
estimates = result.to_estimates_frame()
print(estimates)

# K-dimensional GMM weights per lead:
k_weights = result.to_k_weights_frame()
print(k_weights)
```

### Adding Covariates and Interactions

```python
import numpy as np
import pandas as pd
from diddesign import did

np.random.seed(55)
n_units = 60
times = [0, 1, 2]
units = np.repeat(range(n_units), len(times))
time_col = np.tile(times, n_units)
treated = units < n_units // 2
post = time_col == 2

# Continuous covariates and categorical covariate
x1 = np.random.normal(0, 1, n_units * len(times))
x2 = np.random.normal(0, 1, n_units * len(times))
region = np.random.choice(["A", "B", "C"], n_units * len(times))
y = 3.0 + 0.5 * x1 + 2.0 * (treated & post) + np.random.normal(0, 1, n_units * len(times))

data = pd.DataFrame({
    "unit": units, "time": time_col,
    "treat": (treated & post).astype(int),
    "y": y, "x1": x1, "x2": x2, "region": region,
})

# Covariates: x1*x2 expands to main effects + interaction; factor() for categorical
result = did(data, outcome="y", treatment="treat", time="time",
             unit_id="unit", covariates=["x1*x2", "factor(region)"],
             n_boot=50, random_seed=55)

print(result.to_estimates_frame())
```

### Visualization

```python
from diddesign import did, did_check, plot_estimates, plot_diagnostics

# Assuming `data` is prepared as in the panel example above
result = did(data, formula="y ~ treat", time="time", unit_id="unit",
             n_boot=50, random_seed=42)

check_result = did_check(data=data, outcome="y", treatment="treat",
                         time="time", unit_id="unit",
                         lag=1, n_boot=50, random_seed=42)

# Event-study style plot with placebo overlay
plot_estimates(result, check_fit=check_result,
              title="Double DID Estimates", save="estimates.png", show=False)

# Multi-panel diagnostic figure (trends + placebo)
plot_diagnostics(check_result, result=result,
                 title="Pre-treatment Diagnostics", save="diagnostics.png",
                 show=False)
```

## API Reference

### Public API Summary

- `did()` fits a DID or staggered-adoption design and returns a `DidResult`.
- `did_check()` computes pre-treatment diagnostics and returns a
  `DidCheckResult`.
- `summary()` and `format_summary()` report fitted estimates.
- `fit(..., as_frame=True)` returns event-time rows for figures.
- `check(..., as_frame=True)` returns diagnostic plotting rows.

`DidResult` provides table-ready accessors:

```python
result.to_estimates_frame()
result.to_bootstrap_frame()
result.to_weights_frame()
result.to_gmm_frame()
```

For scripts that need a detached serialized record, `DidResult` and
`DidCheckResult` provide `to_serialized_result()`. New reporting code should
usually start from the frame accessors above because those preserve the table
rows used in the manuscript.

For direct result-object construction, `DidGmmRow` is the public row class for
per-lead GMM matrix entries. `DidCheckResult.named_plot_rows()` returns named
plotting records for diagnostic figures.

### `did()`

Fit a DID or staggered-adoption design and return a `DidResult`.

```python
did(data, *, formula=None, outcome=None, treatment=None, time,
    unit_id=None, post=None, design="did", data_type="panel",
    covariates=None, lead=0, thres=None, n_boot=30, se_boot=None,
    level=95, id_cluster=None, random_seed=None, parallel=False,
    n_cores=None, option=None, is_panel=None, kmax=2, jtest=False)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | DataFrame | — | Input data (panel or repeated cross-section) |
| `formula` | str \| None | `None` | R-style formula, e.g. `"y ~ treat"` or `"y ~ treat + post \| x1 + factor(x2)"` |
| `outcome` | str \| None | `None` | Outcome column name (alternative to formula) |
| `treatment` | str \| None | `None` | Treatment indicator column |
| `time` | str | — | Time period column (required) |
| `unit_id` | str \| None | `None` | Unit identifier column (required for panel) |
| `post` | str \| None | `None` | Post-treatment indicator (required for RCS) |
| `design` | str | `"did"` | `"did"` for standard or `"sa"` for staggered adoption |
| `data_type` | str | `"panel"` | `"panel"` or `"rcs"` (repeated cross-section) |
| `covariates` | list[str] \| None | `None` | Covariate terms: `"x1"`, `"factor(x2)"`, `"x1:x2"`, `"x1*x2"` |
| `lead` | int \| list[int] | `0` | Lead(s) for staggered adoption |
| `thres` | int \| None | `None` | Minimum observations threshold |
| `n_boot` | int | `30` | Number of bootstrap replications |
| `se_boot` | bool \| None | `None` | Use bootstrap percentile CI (`None` = asymptotic for DID, bootstrap for SA) |
| `level` | int | `95` | Confidence level (50–99) |
| `id_cluster` | str \| None | `None` | Cluster variable for clustered bootstrap |
| `random_seed` | int \| None | `None` | Seed for reproducibility |
| `parallel` | bool | `False` | Enable parallel bootstrap computation |
| `n_cores` | int \| None | `None` | Number of cores (default: all available) |
| `kmax` | int | `2` | Maximum DID order: 2 = Double DID, ≥ 3 = K-DID |
| `jtest` | bool | `False` | Apply J-test overidentification moment selection for K-DID |

**Returns:** `DidResult`

### `did_check()`

Compute pre-treatment diagnostic tests. All parameters are keyword-only.

```python
did_check(*, data=None, formula=None, outcome=None, treatment=None,
          time=None, unit_id=None, post=None, design="did",
          covariates=None, data_type="panel", id_cluster=None,
          lag=1, thres=None, n_boot=30, random_seed=None, option=None,
          is_panel=None)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `data` | DataFrame | — | Input data |
| `lag` | int \| list[int] | `1` | Pre-treatment lag(s) to test |
| Other parameters | — | — | Same as `did()` (without `lead`, `kmax`, `jtest`, `parallel`) |

**Returns:** `DidCheckResult`

### `summary()` and `format_summary()`

```python
summary(result, estimator=None, *, as_frame=False)  # → tuple | DataFrame
format_summary(result, estimator=None, *, digits=4)  # → str
```

### `fit()` and `check()`

Data-layer functions for plotting rows:

```python
fit(result, check_fit=None, *, as_frame=False)  # → tuple | DataFrame
check(result, *, as_frame=False)                # → dict
```

### Visualization Functions

All plotting functions require `diddesign[plot]` (matplotlib).

| Function | Input | Description |
| --- | --- | --- |
| `plot_estimates(result, *, check_fit, ...)` | `DidResult` | Event-study plot with optional placebo overlay |
| `plot_trends(check_result, *, ci, ...)` | `DidCheckResult` | Pre-treatment trend comparison |
| `plot_placebo(check_result, *, ...)` | `DidCheckResult` | Placebo estimate plot |
| `plot_pattern(check_result, *, ...)` | `DidCheckResult` | Staggered-adoption pattern diagnostic |
| `plot_diagnostics(check_result, *, result, panels, ...)` | `DidCheckResult` | Multi-panel diagnostic figure |

**Common parameters:**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `title` | str \| None | `None` | Figure title |
| `xlabel` | str \| None | `None` | X-axis label |
| `ylabel` | str \| None | `None` | Y-axis label |
| `figsize` | tuple | `(8, 5)` | Figure size in inches |
| `style` | str | `"publication"` | `"publication"` or `"default"` |
| `save` | str \| None | `None` | File path to save figure |
| `dpi` | int | `150` | Resolution for saved figure |
| `ax` | Axes \| None | `None` | Pre-existing matplotlib axes |
| `show` | bool | `True` | Call `plt.show()` |

### Result Objects

#### `DidResult`

Immutable result object returned by `did()`.

| Method | Returns | Description |
| --- | --- | --- |
| `estimate_rows()` | tuple[DidEstimateRow, ...] | All component and combined estimate rows |
| `to_estimates_frame()` | DataFrame | Estimates as pandas DataFrame |
| `to_bootstrap_frame()` | DataFrame | Bootstrap draws (iterations × components) |
| `to_weights_frame()` | DataFrame | GMM weight rows by lead |
| `to_gmm_frame()` | DataFrame | Full GMM calculation rows (covariances, weight matrix) |
| `to_serialized_result()` | dict | Serializable representation for export |
| `.metadata` | dict | Design metadata (time order, column roles, etc.) |

#### `DidCheckResult`

Diagnostic result object returned by `did_check()`.

| Method | Returns | Description |
| --- | --- | --- |
| `to_summary_frame()` | DataFrame | Placebo test summary |
| `to_placebo_frame()` | DataFrame | Placebo plotting rows |
| `to_trends_frame()` | DataFrame | Trend comparison rows |
| `to_pattern_frame()` | DataFrame | Staggered-adoption pattern rows |
| `named_plot_rows()` | dict | Named diagnostic plotting records |

#### Row Data Classes

| Class | Fields | Used In |
| --- | --- | --- |
| `DidEstimateRow` | estimator, lead, estimate, std_error, ci_lo, ci_hi, weight | `to_estimates_frame()` |
| `DidBootstrapDraw` | iteration, lead, did, sdid | `to_bootstrap_frame()` |
| `DidBootstrapDrawK` | iteration, lead, component_1, component_2, ... | K-DID bootstrap |
| `DidWeightRow` | lead, w_did, w_sdid, double_did_available | `to_weights_frame()` |
| `DidGmmRow` | lead, vcov_*, W_*, gmm_variance | `to_gmm_frame()` |
| `DidCheckDiagnosticRow` | lag, estimate, std_error, ... | Placebo diagnostics |
| `DidCheckTrendRow` | time, treated_mean, control_mean, ... | Trend comparison |
| `DidCheckPatternRow` | lead, lag, estimate, ... | SA pattern diagnostics |

## Methodology

### GMM Combination (Double DID)

The Double DID estimator combines DID ($\hat{\tau}_{DID}$) and sequential DID
($\hat{\tau}_{sDID}$) using efficient GMM weights:

$$\hat{\tau}_{DDID} = w_{DID} \cdot \hat{\tau}_{DID} + w_{sDID} \cdot \hat{\tau}_{sDID}$$

where the weights minimize the asymptotic variance:

$$w = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}' \Sigma^{-1} \mathbf{1}}$$

and $\Sigma$ is the bootstrap covariance matrix of
$(\hat{\tau}_{DID}, \hat{\tau}_{sDID})'$.

### Bootstrap Inference

Standard errors and confidence intervals are obtained via a nonparametric
bootstrap (cluster bootstrap when `id_cluster` is specified):

1. Resample units (or clusters) with replacement.
2. Re-estimate DID and sDID on each bootstrap sample.
3. Estimate the covariance matrix $\hat{\Sigma}$ from bootstrap draws.
4. Compute GMM-optimal weights and the combined Double DID estimate.
5. Confidence intervals use either the asymptotic normal approximation
   (default for `design="did"`) or percentile bootstrap (`se_boot=True`
   or `design="sa"`).

### K-DID Extension

For panels with $K \geq 3$ pre-treatment periods, higher-order transformed-
outcome estimators provide additional moment conditions. The K-DID estimator
combines all $K$ component estimators:

$$\hat{\tau}_{K\text{-}DID} = \mathbf{w}' \hat{\boldsymbol{\tau}}_K$$

where $\hat{\boldsymbol{\tau}}_K = (\hat{\tau}_1, \ldots, \hat{\tau}_K)'$
and weights are chosen by efficient GMM from the $K \times K$ bootstrap
covariance matrix.

### J-test Moment Selection

When `jtest=True`, the overidentification test statistic is used to
adaptively select which moment conditions (component estimators) to retain.
Components that fail the J-test are excluded before the final GMM combination,
providing robustness against misspecified higher-order identification conditions.

### Staggered Adoption

For designs with multiple treatment-timing groups, the staggered-adoption
estimator (`design="sa"`) computes lead-specific SA-DID, SA-sDID, and
SA-Double-DID estimates. Each lead $\ell$ uses units treated at time $g$ and
compares their outcome at $g + \ell$ to never-treated units, applying the
same GMM combination within each lead.

## Reproducing the Paper

To reproduce the article outputs from the repository root, run:

```bash
bash paper/build.sh
```

Set `PYTHON=/path/to/python` to choose a specific interpreter.

## Citation

If you use `diddesign` in your research, please cite the original methodology
paper:

```bibtex
@article{egami2023double,
  title   = {Double Difference-in-Differences},
  author  = {Egami, Naoki and Yamauchi, Soichiro},
  journal = {Political Analysis},
  year    = {2023},
  doi     = {10.1017/pan.2023.8}
}
```

## References

- Egami, N. and Yamauchi, S. (2023). "Double Difference-in-Differences."
  *Political Analysis*. DOI: 10.1017/pan.2023.8
- R package: [DIDdesign](https://github.com/naoki-egami/DIDdesign) (CRAN)
- Stata package: [diddesign](https://github.com/gorgeousfish/diddesign)

## License

GPL-2.0. See [LICENSE](LICENSE) for details.
