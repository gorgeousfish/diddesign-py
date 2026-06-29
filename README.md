# diddesign

**Double Difference-in-Differences for Python**

[![Python ≥3.12](https://img.shields.io/badge/python-%E2%89%A53.12-blue)](https://www.python.org/)
[![License: GPL-2.0](https://img.shields.io/badge/license-GPL--2.0-green)](LICENSE)

<p align="center">
  <img src="image/image.png" alt="diddesign" width="100%">
</p>

## Overview

`diddesign` implements the Double Difference-in-Differences estimator
proposed by Egami and Yamauchi (2023, *Political Analysis*). The method
addresses a practical question in observational panel studies: when multiple
pre-treatment periods are available, how should an analyst exploit them to
strengthen causal inference?

The standard DID estimator requires parallel trends between the last
pre-treatment period and the post-treatment period. The sequential DID
(sDID) estimator requires the weaker parallel trends-in-trends assumption
but uses an additional pre-treatment period. These two assumptions are
logically distinct—neither implies the other. The Double DID formulates a
GMM problem that combines both estimators with variance-minimizing weights.
When both identification conditions hold, the combined estimate achieves
lower variance than either component alone; when only one holds, the
framework remains consistent under that condition.

The package extends to K-DID (Appendix E of the paper): given K ≥ 3
pre-treatment periods, higher-order transformed-outcome estimators provide
additional moment conditions that permit up to (K−1)-degree polynomial
time-varying confounding, combined via the same GMM weighting. A J-test
moment-selection step adaptively discards misspecified components.

For staggered-adoption designs, the package computes lead-specific SA-DID,
SA-sDID, and SA-Double-DID (or SA-K-DID) estimates and aggregates them into
a time-weighted average treatment effect.

## Identification Assumptions

The Double DID framework rests on two assumptions about the counterfactual
trend structure:

| Assumption | Formal Name               | Permitted Bias               |
| ---------- | ------------------------- | ---------------------------- |
| A1         | Parallel Trends           | Constant between groups      |
| A2         | Parallel Trends-in-Trends | Linear change between groups |

Standard DID requires A1. Sequential DID requires A2. The Double DID
requires only that at least one of A1 or A2 holds—a strictly weaker
condition than either assumption alone. With K ≥ 3 pre-treatment periods,
K-DID adds higher-order assumptions that accommodate polynomial confounding.

The identifying relationship is:

```
Extended Parallel Trends
        ↓ implies both
   ┌────┴────┐
   ↓         ↓
Standard    Parallel
Parallel    Trends-in-
Trends      Trends
```

## Installation

```bash
pip install diddesign
```

For repository contributors (editable install from the repo root):

```bash
python3 -m pip install -e diddesign-py
```


Visualization support (matplotlib):

```bash
pip install "diddesign[plot]"
```

**Requirements:** Python ≥ 3.12, NumPy ≥ 1.26, pandas ≥ 2.2.

## Usage

The package provides three entry points:

- `did()` fits a DID or staggered-adoption design and returns a `DidResult`.
- `did_check()` computes pre-treatment diagnostics and returns a `DidCheckResult`.
- `DidResult` provides table-ready accessors (`to_dataframe()`, `to_weights_frame()`, `to_gmm_frame()`, `to_latex()`).

The Double DID workflow proceeds in two steps: first assess the plausibility
of identification assumptions via pre-treatment diagnostics, then estimate
treatment effects conditional on those diagnostics.

### Step 1: Assess Pre-treatment Assumptions

```python
from diddesign.data import data
from diddesign import did_check

df = data("malesky2014")

check = did_check(
    data=df, outcome="pro4", treatment="treatment",
    time="year", post="post_treat", data_type="rcs",
    id_cluster="id_district", lag=[1], n_boot=50, random_seed=1234,
)
print(check.to_summary_frame())
```

Output:

```text
   lag  estimate_raw  std_error_raw  eqci95_lb_std  eqci95_ub_std
0    1      -0.00337       0.041026      -0.163403       0.163403
```

The placebo estimates test whether the DID and sDID estimators yield
approximately zero effects in pre-treatment periods where no effect should
exist. The equivalence confidence interval (reported in units of baseline
control-group standard deviations) provides positive evidence for approximate
parallel trends when it excludes substantively large deviations. No universal
cutoff exists; researchers must apply domain knowledge.

### Step 2: Estimate Treatment Effects

```python
from diddesign import did

result = did(
    df, outcome="pro4", treatment="treatment",
    time="year", post="post_treat", data_type="rcs",
    id_cluster="id_district", n_boot=200, random_seed=1234,
)
print(result.to_dataframe())
print(result.to_latex(caption="Recentralization Effect on Pro4"))
```

Output:

```text
    estimator  lead  estimate  std_error     ci_lo     ci_hi    weight
0  Double-DID     0  0.076596   0.046146 -0.013849  0.167041       NaN
1         DID     0  0.079314   0.057338 -0.033066  0.191694  1.806658
2        sDID     0  0.082684   0.089100 -0.091949  0.257317 -0.806658

\begin{table}[htbp]
\centering
\caption{Recentralization Effect on Pro4}
\begin{tabular}{llrrrr}
\hline\hline
Estimator & Lead & Estimate & Std. Error & CI Low & CI High \\
\hline
Double-DID & 0 & 0.0766$^{*}$ & 0.0461 & -0.0138 & 0.1670 \\
DID & 0 & 0.0793 & 0.0573 & -0.0331 & 0.1917 \\
sDID & 0 & 0.0827 & 0.0891 & -0.0919 & 0.2573 \\
\hline\hline
\multicolumn{6}{l}{\footnotesize Note: $^{*}$p$<$0.10, $^{**}$p$<$0.05, $^{***}$p$<$0.01} \\
\end{tabular}
\end{table}
```

The output reports three rows: Double-DID (the GMM-optimal combination), DID,
and sDID. The GMM weights indicate which component the data favor—when
`w_DID ≈ 1`, standard DID dominates; when `w_sDID ≈ 1`, the sequential
component dominates.

### Staggered Adoption

```python
import numpy as np
from diddesign.data import data
from diddesign import did, did_check

df = data("paglayan2019")
df["log_expenditure"] = np.log(df["pupil_expenditure"] + 1.0)

# Diagnose pre-trends across multiple lags
check = did_check(
    data=df, outcome="log_expenditure", treatment="treatment",
    time="year", unit_id="state", design="sa",
    lag=[1, 2, 3], thres=1, n_boot=50, random_seed=1234,
)
print(check.to_summary_frame())

# SA-Double-DID estimation
result = did(
    df, outcome="log_expenditure", treatment="treatment",
    time="year", unit_id="state", design="sa",
    thres=1, n_boot=200, random_seed=1234,
)
print(result.to_dataframe())
```

Output:

```text
   lag  estimate_raw  std_error_raw  eqci95_lb_std  eqci95_ub_std
0    1     -0.002669       0.009736      -0.117499       0.117499
1    2     -0.012447       0.007841      -0.151357       0.151357
2    3      0.002269       0.011331      -0.121691       0.121691

       estimator  lead  estimate  std_error     ci_lo     ci_hi    weight
0  SA-Double-DID     0  0.011401   0.012157 -0.011430  0.033800       NaN
1         SA-DID     0  0.010984   0.012247 -0.011420  0.034097  0.843723
2        SA-sDID     0  0.013653   0.014537 -0.014634  0.042717  0.156277
```

### K-DID with J-test Moment Selection

When three or more pre-treatment periods are available, K-DID exploits
higher-order moment conditions. The J-test adaptively removes components
whose identifying assumptions appear violated.

```python
result = did(
    df, outcome="log_expenditure", treatment="treatment",
    time="year", unit_id="state", design="sa",
    kmax=3, jtest=True, thres=1,
    n_boot=200, random_seed=1234,
)
print(result.to_dataframe())
```

Output:

```text
   estimator  lead  estimate  std_error     ci_lo     ci_hi weight
0   SA-K-DID     0  0.011685   0.012156 -0.011180  0.034105   None
1     SA-DID     0  0.010984   0.012247 -0.011420  0.034097   None
2    SA-sDID     0  0.013653   0.014537 -0.014634  0.042717   None
3  SA-kDID-3     0  0.003875   0.023613 -0.040192  0.052995   None
```

### Covariates and Interactions

The package supports continuous covariates, `factor()` categorical encoding,
and `x1:x2` interaction terms (or `x1*x2` for main effects plus interaction):

```python
result = did(
    data, outcome="y", treatment="treat", time="time",
    unit_id="unit", covariates=["x1*x2", "factor(region)"],
    n_boot=50, random_seed=55,
)
```

### Visualization

```python
from diddesign import plot_estimates, plot_diagnostics

plot_estimates(result, check_fit=check,
              title="Double DID Estimates", save="estimates.png", show=False)

plot_diagnostics(check, result=result,
                 title="Pre-treatment Diagnostics", save="diagnostics.png",
                 show=False)
```

## Methodology

The Double DID estimator combines DID (τ̂_DID) and sequential DID (τ̂_sDID)
using efficient GMM weights:

$$
\hat{\tau}_{DDID} = w_{DID} \cdot \hat{\tau}_{DID} + w_{sDID} \cdot \hat{\tau}_{sDID}
$$

where the weights minimize asymptotic variance:

$$
w = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}' \Sigma^{-1} \mathbf{1}}
$$

and Σ is the bootstrap covariance matrix of (τ̂_DID, τ̂_sDID)'. Under the
extended parallel trends assumption:

$$
\text{Var}(\hat{\tau}_{DDID}) \leq \min\{\text{Var}(\hat{\tau}_{DID}),\; \text{Var}(\hat{\tau}_{sDID})\}
$$

Standard errors and confidence intervals are obtained via a nonparametric
bootstrap (cluster bootstrap when `id_cluster` is specified). The bootstrap
covariance matrix Σ̂ determines the GMM-optimal weights, and percentile or
normal-approximation intervals are reported depending on the design.

For K-DID, the K-dimensional generalization combines all K component
estimators via the K×K bootstrap covariance, with the J-test providing a
model-selection step that excludes moments whose overidentification statistic
rejects.

## API Reference

### Core Functions

| Function                      | Purpose                                              |
| ----------------------------- | ---------------------------------------------------- |
| `did()`                     | Fit DID or staggered-adoption design →`DidResult` |
| `did_check()`               | Pre-treatment diagnostics →`DidCheckResult`       |
| `summary()`                 | Formatted summary of fitted result                   |
| `fit(..., as_frame=True)`   | Event-time plotting rows                             |
| `check(..., as_frame=True)` | Diagnostic plotting rows                             |

### `did()` Parameters

```python
did(data, *, formula=None, outcome=None, treatment=None, time,
    unit_id=None, post=None, design="did", data_type="panel",
    covariates=None, lead=0, thres=None, n_boot=30, se_boot=None,
    level=95, id_cluster=None, random_seed=None, parallel=False,
    n_cores=None, parallel_backend="thread", worker_timeout=None,
    verbose=1, kmax=2, jtest=False)
```

| Parameter       | Type             | Default     | Description                                                          |
| --------------- | ---------------- | ----------- | -------------------------------------------------------------------- |
| `data`        | DataFrame        | —          | Input data (panel or repeated cross-section)                         |
| `formula`     | str\| None       | `None`    | R-style formula, e.g.`"y ~ treat"`                                 |
| `outcome`     | str\| None       | `None`    | Outcome column name (alternative to formula)                         |
| `treatment`   | str\| None       | `None`    | Treatment indicator column                                           |
| `time`        | str              | —          | Time period column (required)                                        |
| `unit_id`     | str\| None       | `None`    | Unit identifier column (required for panel)                          |
| `post`        | str\| None       | `None`    | Post-treatment indicator (required for RCS)                          |
| `design`      | str              | `"did"`   | `"did"` for standard or `"sa"` for staggered adoption            |
| `data_type`   | str              | `"panel"` | `"panel"` or `"rcs"` (repeated cross-section)                    |
| `covariates`  | list[str]\| None | `None`    | Covariate terms:`"x1"`, `"factor(x2)"`, `"x1:x2"`, `"x1*x2"` |
| `lead`        | int\| list[int]  | `0`       | Lead(s) for staggered adoption                                       |
| `thres`       | int\| None       | `None`    | Minimum observations threshold                                       |
| `n_boot`      | int              | `30`      | Number of bootstrap replications                                     |
| `se_boot`     | bool\| None      | `None`    | Use bootstrap percentile CI                                          |
| `level`       | int              | `95`      | Confidence level (50–99)                                            |
| `id_cluster`  | str\| None       | `None`    | Cluster variable for clustered bootstrap                             |
| `random_seed` | int\| None       | `None`    | Seed for reproducibility                                             |
| `parallel`    | bool             | `False`   | Enable parallel bootstrap computation                                |
| `n_cores`     | int\| None       | `None`    | Number of cores (default: all available)                             |
| `kmax`        | int              | `2`       | Maximum DID order: 2 = Double DID, ≥ 3 = K-DID                      |
| `jtest`       | bool             | `False`   | Apply J-test moment selection for K-DID                              |

**Returns:** `DidResult`

### `did_check()` Parameters

```python
did_check(*, data=None, formula=None, outcome=None, treatment=None,
          time=None, unit_id=None, post=None, design="did",
          covariates=None, data_type="panel", id_cluster=None,
          lag=1, thres=None, n_boot=30, random_seed=None,
          verbose=1)
```

All parameters are keyword-only. Parameters shared with `did()` have
identical semantics. The `lag` parameter specifies which pre-treatment
lag(s) to test.

**Returns:** `DidCheckResult`

### Result Objects

`DidResult` is an immutable object returned by `did()`. It provides frame
accessors for downstream analysis:

```python
result.to_dataframe()           # Estimates as DataFrame
result.to_estimates_frame()     # Alias
result.to_bootstrap_frame()     # Bootstrap draws (iterations × components)
result.to_weights_frame()       # GMM weight rows by lead
result.to_gmm_frame()          # Full GMM calculation rows
result.to_k_weights_frame()    # K-dimensional GMM weights (K-DID)
result.to_latex()              # LaTeX table string
result.to_serialized_result()  # Serializable dict for export
```

`DidCheckResult` is an immutable object returned by `did_check()`:

```python
check.to_summary_frame()   # Placebo test summary
check.to_placebo_frame()   # Placebo plotting rows
check.to_trends_frame()    # Trend comparison rows
check.to_pattern_frame()   # SA pattern rows
check.named_plot_rows()    # Named plotting records
```

Each row in the GMM frame is a `DidGmmRow` containing the covariance entries,
weights, and GMM variance for a single lead. The diagnostic result provides
`DidCheckResult.named_plot_rows()` for downstream figure production.

For scripts that need a detached serialized record, `DidResult` and
`DidCheckResult` provide `to_serialized_result()`. New reporting code should
usually start from the frame accessors above because those preserve the table
rows used in the manuscript.

### Data Loading

```python
from diddesign.data import data

df = data("malesky2014")    # Vietnam RCS (Malesky et al. 2014)
df = data("paglayan2019")   # US states panel (Paglayan 2019)
```

### Visualization Functions

All plotting functions require `diddesign[plot]` (matplotlib).

| Function                                | Input              | Description                                    |
| --------------------------------------- | ------------------ | ---------------------------------------------- |
| `plot_estimates(result, ...)`         | `DidResult`      | Event-study plot with optional placebo overlay |
| `plot_trends(check_result, ...)`      | `DidCheckResult` | Pre-treatment trend comparison                 |
| `plot_placebo(check_result, ...)`     | `DidCheckResult` | Placebo estimate plot                          |
| `plot_pattern(check_result, ...)`     | `DidCheckResult` | Staggered-adoption pattern diagnostic          |
| `plot_diagnostics(check_result, ...)` | `DidCheckResult` | Multi-panel diagnostic figure                  |

### Errors

`diddesign` provides structured exceptions with machine-readable error codes
(E001–E020) and diagnostic context dictionaries:

```python
from diddesign.errors import DidValueError, ErrorCode

try:
    result = did(data=df, outcome="y", treatment="bad_col", time="t", unit_id="id")
except DidValueError as e:
    print(e.code)     # ErrorCode.E001
    print(e.context)  # {'field_name': 'treatment', ...}
```

Full parameter documentation is available in the
[Sphinx API reference](https://diddesign.readthedocs.io/en/latest/api.html).
To build docs locally: `cd Docs && sphinx-build -b html . _build/html`.

## Citation

If you use this package in your research, please cite both the software and the methodology paper:

**APA Format:**

> Xu, W. (2025). *diddesign: Python package for Double Difference-in-Differences estimation* (Version 0.1.0) [Computer software]. GitHub. https://github.com/gorgeousfish/diddesign

> Egami, N., & Yamauchi, S. (2023). Using Multiple Pretreatment Periods to Improve Difference-in-Differences and Staggered Adoption Designs. *Political Analysis*, 31(2), 195-212. https://doi.org/10.1017/pan.2022.8

**BibTeX:**

```bibtex
@software{diddesign2025python,
  title={diddesign: Python package for Double Difference-in-Differences estimation},
  author={Wenli Xu},
  year={2025},
  version={0.1.0},
  url={https://github.com/gorgeousfish/diddesign}
}

@article{egami2023using,
  title={Using Multiple Pretreatment Periods to Improve Difference-in-Differences and Staggered Adoption Designs},
  author={Egami, Naoki and Yamauchi, Soichiro},
  journal={Political Analysis},
  volume={31},
  number={2},
  pages={195--212},
  year={2023},
  doi={10.1017/pan.2022.8}
}
```

## Authors

**Python Implementation:**

- **Xuanyu Cai**, City University of Macau
  Email: [xuanyuCAI@outlook.com](mailto:xuanyuCAI@outlook.com)
- **Wenli Xu**, City University of Macau
  Email: [wlxu@cityu.edu.mo](mailto:wlxu@cityu.edu.mo)

**Methodology:**

- **Naoki Egami**, Columbia University
- **Soichiro Yamauchi**, Harvard University

## See Also

- R package by Egami & Yamauchi: [DIDdesign](https://github.com/naoki-egami/DIDdesign)
- Stata package: [diddesign](https://github.com/gorgeousfish/diddesign)
- Paper: [10.1017/pan.2022.8](https://doi.org/10.1017/pan.2022.8)

## License

GPL-2.0. See [LICENSE](LICENSE) for details.
