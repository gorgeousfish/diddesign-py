# diddesign examples

These examples show the package workflow used by the manuscript without relying
on manuscript-only formatting code. Run them from the repository root after the
package dependencies are available:

```bash
python3 diddesign-py/examples/malesky_rcs_workflow.py
python3 diddesign-py/examples/paglayan_sa_workflow.py
```

The scripts print compact JSON summaries and write no manuscript outputs. The
summaries include returned-row counts, component estimates, GMM weights,
diagnostic or fitted figure rows with 90% interval endpoints, and a small
recomposition check for the reported Double-DID rows. They demonstrate the
public calls that the paper evaluates:

- `did()` returns a `DidResult` with estimate, bootstrap, weight, and GMM frames.
- `did_check()` returns a `DidCheckResult` with diagnostic frames.
- `fit(..., as_frame=True)` returns plotting rows before any figure is
  rendered.

The public scripts keep bootstrap counts small enough for a quick package
check: the Malesky workflow uses `n_boot=20`, and the Paglayan workflow uses
`n_boot=4`. The manuscript generators rerun the same public calls with a larger
bootstrap setting for the displayed interval figures, so these examples should
be read as workflow checks rather than standalone inferential evidence.

Read the JSON summaries in the same order used by the manuscript examples. In
the Malesky repeated-cross-section workflow, inspect `component_estimates` and
`gmm_weights` before reporting `double_did_estimate`. A non-convex weight pair
means the combined row is an extrapolating GMM recomposition rather than a
simple average. The `diagnostic_row` and `fit_rows` fields then show how the
lagged placebo row is placed beside the treatment-period row; each fit row also
includes `ci90_lb` and `ci90_ub`, the interval endpoints used by plotting code.
In the Paglayan staggered-adoption workflow, the estimate, component, weight,
and fit rows are keyed by lead, so each plotted point should be read with the
component and weight rows for the same lead key and with its own `ci90_lb` and
`ci90_ub` values.

The data files are repository examples from the companion R/Stata assets. The
scripts are software examples, not new substantive analyses of the original
studies.
