# diddesign Benchmarks

Performance and numerical parity benchmarks for the Python implementation.

## Running Benchmarks

```bash
# All benchmarks
cd diddesign-py
python -m pytest benchmarks/ -v

# Numerical parity only
python -m pytest benchmarks/test_numerical_parity.py -v

# Performance only (with timing output)
python -m pytest benchmarks/test_performance.py -v -s
```

## Tolerance Standards

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Point estimates | < 1e-8 | Deterministic computation |
| GMM weights | sum = 1 ± 1e-10 | Theoretical: w_DID + w_sDID = 1 |
| Bootstrap SE | CV < 0.5 across seeds | Expected stochastic variation |
| GMM variance | > 0 | Positive definiteness |
| K-DID weights | sum ≈ 1 per lead | Theoretical constraint |

## Structure

- `test_numerical_parity.py` — Verifies point estimate correctness, weight
  constraints, bootstrap stability, K-DID structure, and deterministic
  reproducibility. Stata reference values are placeholder (None) until
  cross-validated.
- `test_performance.py` — Measures execution time for RCS, SA, K-DID
  estimation, diagnostics, n_boot scaling, and result object operations.
  Time limits are conservative to accommodate CI environments.

## Filling Stata Reference Values

Once Stata reference values are available (from `diddesign v0.1.0`,
`seed=1234`, `nboot=200`), update the `MALESKY_RCS_REFERENCE` and
`PAGLAYAN_SA_REFERENCE` dictionaries in `test_numerical_parity.py` and
add assertions comparing Python estimates against them with tolerance < 1e-8.
