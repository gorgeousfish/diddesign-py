User Guide
==========

Installation
------------

From PyPI:

.. code-block:: bash

   pip install diddesign

For visualization support (matplotlib):

.. code-block:: bash

   pip install "diddesign[plot]"

For editable development installs (repository contributors only):

.. code-block:: bash

   python3 -m pip install -e diddesign-py

**Requirements:** Python >= 3.12, NumPy >= 1.26, pandas >= 2.2.

Quick Start
-----------

The package exposes two primary functions:

- :func:`diddesign.did` — estimate treatment effects.
- :func:`diddesign.did_check` — pre-treatment placebo diagnostics.

A typical workflow diagnoses pre-treatment assumptions first, then estimates:

.. code-block:: python

   from diddesign.data import data
   from diddesign import did, did_check, summary

   df = data("malesky2014")

   # Step 1: pre-treatment placebo diagnostic
   check = did_check(
       data=df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district", lag=[1], n_boot=50, random_seed=1234,
   )
   print(check.to_summary_frame())

.. code-block:: text

    lag  estimate_raw  std_error_raw  eqci95_lb_std  eqci95_ub_std
      1      -0.00337       0.041026      -0.163403       0.163403

.. code-block:: python

   # Step 2: estimate treatment effects
   result = did(
       df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district", n_boot=200, random_seed=1234,
   )
   print(summary(result, as_frame=True))

.. code-block:: text

    estimator  lead  estimate  std.error  statistic  p_value    ci.low  ci.high
   Double-DID     0  0.076596   0.046146   1.659849 0.096945 -0.013849 0.167041
          DID     0  0.079314   0.057338   1.383284 0.166578 -0.033066 0.191694
         sDID     0  0.082684   0.089100   0.927990 0.353413 -0.091949 0.257317

Data Requirements
-----------------

The estimator accepts two data structures through the ``data_type`` parameter.

**Panel data** (``data_type="panel"``):
   Repeated observations of the same units over time. Each row is a
   unit-time pair. Required columns:

   - ``outcome``: the dependent variable.
   - ``treatment``: binary (0/1) treatment indicator.
   - ``time``: integer or ordered time period.
   - ``unit_id``: unit identifier (string or integer).

**Repeated cross-sections** (``data_type="rcs"``):
   Different units sampled in each period. Required columns:

   - ``outcome``: the dependent variable.
   - ``treatment``: binary (0/1) treatment indicator.
   - ``time``: integer or ordered time period.
   - ``post``: binary (0/1) indicator for post-treatment periods.
   - ``id_cluster`` (optional): cluster identifier for clustered bootstrap.

Designs
-------

The package supports three estimation designs through the ``design`` and
``kmax`` parameters.

**Standard Double DID** (``design="did"``, ``kmax=2``):
   Two pre-treatment periods. Combines DID and sequential DID via efficient
   GMM weights. This is the default. Accepts both panel and RCS data.

**K-DID** (``design="did"``, ``kmax>=3``):
   Three or more pre-treatment periods. Higher-order moment conditions
   accommodate polynomial confounding. The ``jtest=True`` option applies a
   J-test moment-selection step that drops misspecified components. Requires
   panel data with at least ``kmax`` pre-treatment periods.

**Staggered Adoption** (``design="sa"``):
   Units adopt treatment at different times. Computes lead-specific SA-DID,
   SA-sDID, and SA-Double-DID (or SA-K-DID) estimates, aggregated by
   treatment-timing weights. Requires panel data (``data_type="panel"``).
   The ``thres`` parameter sets the minimum number of treated units per
   adoption cohort.

Design Choice Guide
~~~~~~~~~~~~~~~~~~~

Use ``design="did"`` when:

- Treatment occurs at a common time for all treated units.
- Works with panel or repeated cross-section data.

Use ``design="sa"`` when:

- Units adopt treatment at different times (staggered adoption).
- Requires panel data (``data_type="panel"``).

``kmax`` guidelines:

- ``kmax=2`` (default): recommended for most applications.
- ``kmax>=3``: only when the panel has at least ``kmax`` pre-treatment
  periods and higher-order parallel-trends assumptions are plausible.
- With ``kmax>=3``, set ``jtest=True`` to adaptively remove misspecified
  moment conditions.

Parameter Selection Guide
-------------------------

The following guidance helps choose key parameters for applied work.

``n_boot`` (integer, required):
   Number of bootstrap replications for inference. For exploratory analysis,
   50--200 replications provide quick feedback. For publication-quality
   inference, use at least 1000. Larger values reduce bootstrap noise in
   standard errors and confidence intervals at the cost of computation time.

``random_seed`` (integer or None):
   Seed for the bootstrap random number generator. Set an integer value for
   reproducibility. Omit or pass ``None`` for non-reproducible runs.

``kmax`` (integer, default 2):
   Maximum number of pre-treatment periods to exploit. With ``kmax=2``, the
   estimator uses DID and sequential DID (the standard Double DID). With
   ``kmax=3`` or higher, additional moment conditions enter the GMM problem.
   Increase ``kmax`` only when the panel has enough pre-treatment periods and
   the higher-order parallel-trends assumptions are plausible.

``jtest`` (bool, default False):
   When ``kmax>=3``, setting ``jtest=True`` applies a J-test moment-selection
   procedure that drops components whose identifying assumptions appear
   violated. This guards against bias from misspecified higher-order conditions.

``thres`` (integer, default 0):
   Minimum number of treated units in an adoption cohort for staggered-adoption
   designs. Cohorts below this threshold are excluded. Set ``thres=1`` to
   exclude empty cohorts only.

``se_boot`` (bool or None):
   Controls inference method. With ``None`` (default for ``design="did"``),
   bootstrap identifies the weight matrix and standard errors while confidence
   intervals use the efficient-GMM asymptotic formula. With ``True``, percentile
   bootstrap intervals replace asymptotic ones. Staggered-adoption designs
   require ``se_boot=True`` (set automatically).

``lead`` (list of int or None):
   For staggered-adoption designs, specifies which leads (periods since
   treatment) to estimate. ``None`` estimates all available leads.

``lag`` (list of int):
   For ``did_check``, specifies which pre-treatment lags to test. Each lag
   defines a placebo treatment timing.

Result Objects
--------------

:class:`diddesign.DidResult` is returned by ``did()``. It is immutable and
provides pandas DataFrame accessors:

.. code-block:: python

   result.to_dataframe()           # All estimate rows
   result.to_estimates_frame()     # Alias for to_dataframe()
   result.to_bootstrap_frame()     # Bootstrap draws (iterations x components)
   result.to_weights_frame()       # GMM weight rows by lead
   result.to_gmm_frame()          # Full GMM calculation rows
   result.to_k_weights_frame()    # K-dimensional GMM weights (K-DID)
   result.to_latex()              # LaTeX table output
   result.to_serialized_result()  # Serializable dict for export

:class:`diddesign.DidCheckResult` is returned by ``did_check()``:

.. code-block:: python

   check.to_summary_frame()   # Placebo test summary rows
   check.to_placebo_frame()   # Placebo plotting rows
   check.to_trends_frame()    # Trend comparison rows
   check.to_pattern_frame()   # SA pattern rows
   check.named_plot_rows()    # Named plotting records

Reading Returned Rows
---------------------

The returned DataFrames follow a deliberate reading order before reporting.

For **Double DID**, read the estimate frame first, then the weight and GMM
frames:

.. code-block:: python

   estimates = result.to_estimates_frame()
   weights = result.to_weights_frame()
   gmm_rows = result.to_gmm_frame()

   # Component estimates
   component_rows = estimates.loc[estimates["estimator"].isin(["DID", "sDID"])]
   # Combined Double-DID row
   combined_row = estimates.loc[estimates["estimator"] == "Double-DID"].iloc[0]

The GMM weights show how DID and sDID were combined. A ``w_did`` of 1.81 and
``w_sdid`` of -0.81 indicates the GMM-optimal combination extrapolates beyond
a convex average. This occurs when the covariance structure of component
estimators makes one component informationally redundant for the combined
estimate.

For **staggered-adoption** designs, read by lead. Each lead has its own SA-DID,
SA-sDID, and SA-Double-DID rows, plus a lead-specific weight row:

.. code-block:: python

   # Filter by lead
   lead0 = estimates.loc[estimates["lead"] == 0]
   weights_lead0 = weights.loc[weights["lead"] == 0]

Interpreting Results
--------------------

The ``summary()`` function returns a frame with test statistics and p-values:

.. code-block:: python

   from diddesign import summary
   print(summary(result, as_frame=True))

Key columns:

- **estimate**: the point estimate of the treatment effect (ATT).
- **std.error**: bootstrap or asymptotic standard error.
- **statistic**: z-statistic (estimate / std.error).
- **p_value**: two-sided p-value.
- **ci.low**, **ci.high**: 95% confidence interval bounds.

The **Double-DID** row is the GMM-optimal combination. The **DID** and **sDID**
rows are the individual component estimators. Compare them to understand whether
the combined estimate derives mainly from one component or represents a genuine
efficiency improvement.

For pre-treatment diagnostics, interpret the equivalence confidence interval
(``eqci95_lb_std``, ``eqci95_ub_std``) as the range of standardized placebo
effects consistent with parallel trends. A placebo estimate near zero with a
narrow equivalence interval supports the identifying assumption.

Formula Interface
-----------------

An alternative to specifying column names individually:

.. code-block:: python

   from diddesign import did

   result = did(
       df, formula="pro4 ~ treatment + post_treat | lnarea + lnpopden",
       time="year", data_type="rcs",
       id_cluster="id_district", n_boot=50, random_seed=1234,
   )

Panel formulas use ``outcome ~ treatment | covariates``.
Repeated cross-section formulas use ``outcome ~ treatment + post | covariates``.

Supported covariate terms: bare column names (continuous), ``factor(column)``
for categorical encoding, ``x1:x2`` for interactions, and ``x1*x2`` for main
effects plus interaction.

Covariates
----------

Pass covariates as a list of term strings:

.. code-block:: python

   result = did(
       df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district",
       covariates=["lnarea", "lnpopden"],
       n_boot=50, random_seed=1234,
   )
   print(result.to_dataframe())

.. code-block:: text

    estimator  lead  estimate  std_error     ci_lo    ci_hi    weight
   Double-DID     0  0.083374   0.042437  0.000199 0.166549       NaN
          DID     0  0.086286   0.054630 -0.020786 0.193359  1.864198
         sDID     0  0.089656   0.085487 -0.077895 0.257207 -0.864198

Supported covariate specifications:

- ``"x1"`` — continuous covariate.
- ``"factor(region)"`` — categorical dummy encoding.
- ``"x1:x2"`` — interaction of x1 and x2.
- ``"x1*x2"`` — main effects plus interaction.

Covariates cannot reuse outcome, treatment, time, unit id, or post columns.

Inference
---------

**Standard DID** (``design="did"``):
   By default (``se_boot=None``), bootstrap identifies the GMM weight matrix
   and component standard errors, while confidence intervals use the
   efficient-GMM asymptotic formula. Set ``se_boot=True`` for percentile
   bootstrap intervals.

**Staggered adoption** (``design="sa"``):
   Bootstrap confidence intervals are required. Passing ``se_boot=False``
   is rejected.

The ``n_boot`` parameter controls bootstrap replications (minimum 2). The
``random_seed`` parameter ensures reproducibility (integer in [0, 2^32 - 1]
or None).

Visualization
-------------

Plotting requires the optional ``diddesign[plot]`` dependency (matplotlib).
Functions consume the plotting rows returned by ``fit()`` and ``check()``:

.. code-block:: python

   from diddesign import fit, plot_estimates, plot_diagnostics

   # Render figures directly from result objects
   plot_estimates(result, title="Double DID Estimates", save="fig.png", show=False)
   plot_diagnostics(check, result=result, save="diag.png", show=False)

   # Or prepare plotting rows for custom use
   fit_rows = fit(result, as_frame=True)
   print(fit_rows)

Available plot functions:

- :func:`diddesign.plot_estimates` — event-study plot with optional
  placebo overlay.
- :func:`diddesign.plot_trends` — pre-treatment trend comparison.
- :func:`diddesign.plot_placebo` — placebo estimate plot.
- :func:`diddesign.plot_pattern` — staggered-adoption pattern diagnostic.
- :func:`diddesign.plot_diagnostics` — multi-panel diagnostic figure.

Built-in Datasets
-----------------

.. code-block:: python

   from diddesign.data import data

   df = data("malesky2014")    # Vietnam RCS (Malesky et al. 2014), 6269 x 42
   df = data("paglayan2019")   # US states panel (Paglayan 2019), 2058 x 9

**malesky2014**: Repeated cross-section data on Vietnamese communes, originally
used to study the effects of administrative recentralization on local public
goods provision (Malesky, Nguyen, and Tran 2014). Key columns: ``pro4``
(outcome), ``treatment``, ``year``, ``post_treat``, ``id_district`` (cluster).

**paglayan2019**: Balanced panel of US states from 1959 to 2000, used to study
the effects of collective bargaining laws on teacher salaries and school
expenditure (Paglayan 2019). Key columns: ``pupil_expenditure``,
``teacher_salary``, ``treatment``, ``year``, ``state`` (unit id).

See :mod:`diddesign.data` for full dataset documentation.

Error Handling
--------------

The estimators validate inputs and raise informative exceptions:

- :class:`~diddesign.DidDataError`: column missing, wrong type, or
  panel structure violated.
- :class:`~diddesign.DataContractError`: design contract violation
  (e.g., treatment not absorbing in SA design).

.. code-block:: python

   from diddesign import did, DidDataError

   try:
       result = did(data=df, outcome="y", treatment="treat", time="t")
   except DidDataError as e:
       print(f"Data issue: {e}")

Bootstrap Inference for Staggered Adoption
------------------------------------------

For staggered-adoption designs, bootstrap confidence intervals are required
because the aggregation across treatment timings does not admit a simple
asymptotic formula.  Set ``se_boot=True`` to request percentile bootstrap
confidence intervals.  Python or NumPy boolean scalars are accepted and
numeric or string flags are rejected with an informative error message.
Passing ``se_boot=False`` with ``design="sa"`` is rejected so that
the inference choice remains explicit.

Summary Statistic Edge Cases
----------------------------

A zero standard error with a non-zero estimate is reported as an infinite
signed statistic with zero p-value, correctly reflecting the extreme
z-score rather than producing undefined arithmetic.

Conversely, the unidentified ``0 / 0`` case is reported as ``None`` for both
statistic and p-value, instead of exposing ``NaN`` in public summary rows.

Rows whose standard error is unavailable remain in the summary output with
their estimator, lead, and point estimate intact.  Only
``std.error``, ``statistic``, ``p_value``, ``ci.low``, and ``ci.high`` are
reported as ``None``.

Plotting Row Requirements
-------------------------

The ``fit()`` function requires available Double-DID estimate rows with
standard errors to produce plotting rows.  Diagnostic overlays require
finite raw placebo estimates with non-negative raw standard errors.

For a single-lead result, ``fit()`` requires a Double-DID row with a standard
error.  For multi-lead ``did()`` results, ``fit()`` emits one Double-DID
plotting row per lead that actually has both a Double-DID estimate and
standard error.  For multi-lead ``did()`` results, ``fit`` emits one
Double-DID plotting row per lead with an available Double-DID estimate and
standard error, and keeps ``time_to_treat`` equal to the lead.  Each row
carries the lead-specific estimate and standard error.

Staggered-adoption pattern diagnostics cannot be overlaid onto a non-SA
result; attempting this results in silently dropping the treatment-pattern
rows from the output frame.

Formula Validation
------------------

The formula argument must be a string conforming to the two-sided R-style
syntax.  ``is_panel`` must be a Python or NumPy boolean scalar; truthy
integers and strings are rejected.

The formula parser accepts bare column names and ``factor(column)`` markers
on the right-hand side.  Direct ``DidFormulaSpec`` objects enforce the same
identifier, covariate-term, and interaction-term validation rules;
unsupported terms are rejected with informative error messages.

The same covariate-term validation applies when terms are passed directly
through ``covariates=``; direct covariates must be a sequence of terms, not
a bare string.

Duplicate covariate column names are rejected before ``did()`` and
``did_check()`` proceed, including mixed encodings such as ``x`` plus
``factor(x)``.

Covariates also cannot reuse outcome, treatment, time, unit id, or post role
columns; those role columns are reserved for the DID design itself.

Time Label Validation
---------------------

Time labels must be all numeric or all string labels.  Numeric time labels
may use Python or numpy real scalars (int, float, numpy integer/float types).
``NaN`` and infinite time values are rejected before panel, repeated
cross-section, or staggered-adoption estimators derive treatment timing.

Boolean Numeric Validation
--------------------------

Manual ``did_check`` diagnostic and plotting rows also treat booleans as
invalid numeric values, rather than coercing ``True`` or ``False`` into
``1.0`` or ``0.0``.

GMM matrix entries must be finite numeric values, not booleans.

Random Seed Validation
----------------------

``random_seed`` must be ``None`` or an integer between ``0`` and ``2**32 - 1``;
booleans, floats, strings, negative values, and out-of-range values are
rejected before any bootstrap sampling starts.

Design Role and Cluster Validation
----------------------------------

``design`` and ``data_type`` must be strings from the supported public choices
(``"did"``, ``"sa"`` for design; ``"panel"``, ``"rcs"`` for data_type).

``outcome``, ``treatment``, ``time``, ``unit_id``, and ``post`` must be
non-empty column name strings when required or provided.  Those design-role
columns must be distinct; any overlap is rejected before any estimator or
diagnostic design matrix is built.

``id_cluster`` must be ``None`` or a non-empty column name string; blank
strings and non-string values are rejected to prevent silently changing the
cluster mode.  Explicit cluster columns may match ``unit_id`` (cluster at the
unit level), but they may not reuse outcome, treatment, time, or post
columns, because those columns define the estimand rather than an
independent clustering variable.

Bootstrap and Sequence Validation
---------------------------------

``n_boot`` must be an integer greater than or equal to 2 for both ``did()``
and data-driven ``did_check()``; booleans, floats, strings, and missing
values are rejected before any bootstrap or diagnostic resampling starts.

``lead`` sequences for ``did()`` and ``lag`` sequences for data-driven
``did_check()`` must contain at least one non-negative integer.

GMM Availability Consistency
----------------------------

GMM rows require ``double_did_available`` to match the availability of both
component weights and a positive GMM variance; the result cannot mark a
complete weighting calculation as unavailable.

``double_did_available`` flags accept Python or NumPy boolean scalars;
numeric and string flag values are rejected.

``double_did_available_leads`` must exactly match the estimate leads for
which both component weights exist.  The global ``double_did_available`` flag
must match whether every estimate lead has a Double-DID row; attempts to
under-report or over-report Double-DID availability are rejected.

``identified_leads`` must match the estimate leads that passed identification.
``filtered_leads`` must match ``unidentified_leads``.
``requested_leads`` must equal the disjoint union of identified and filtered
leads.  Singular alias fields such as ``requested_lead`` and
``identified_lead`` must match their plural fields.

Negative GMM Weights
--------------------

Double-DID efficient-GMM weights are not constrained to be convex weights;
negative component weights are retained when the bootstrap covariance matrix
is finite, symmetric positive definite and the normalized weights sum to one.
These weights define the Double-DID point estimate and efficient-GMM standard
error.

Public ``summary()``, ``fit()``, ``to_estimates_frame()``, and serializers
expose that resulting Double-DID row directly, rather than reinterpreting it
as a convex average.

``weights_by_lead`` must match the normalized row-sum weights implied by
``W_by_lead``; the result cannot expose mutually inconsistent GMM weights
and inverse covariance matrices.  This check applies to both the basic DID
design and the staggered-adoption SA design.

Boolean ``as_frame`` Validation
-------------------------------

The ``as_frame`` option on ``summary()``, ``fit()``, and ``check()`` must be
a Python or NumPy boolean scalar; truthy strings, integers, and other
non-boolean values are rejected to prevent silently switching the return type.

Estimator Filter Validation
---------------------------

The ``estimator`` filter must be a supported estimator string or a non-empty
sequence of supported estimator strings; dictionaries, generators, empty
sequences, and mixed-type sequences are rejected.

Sequential DID Transformed Outcomes
-----------------------------------

Sequential DID transformed outcomes follow Appendix C.3 of Egami and
Yamauchi (2023).  For both panel and repeated cross-section data, sDID uses
group-specific lagged means to construct the transformed outcome: the
current outcome minus the treated group's lagged mean for treated
observations, and the current outcome minus the control group's lagged mean
for control observations.

SA Component Timing Weights
---------------------------

SA time-average components use treatment-timing weights to aggregate across
adoption cohorts.  When a lead/timing path is unavailable for one component
but available for another, SA-DID and SA-sDID drop unavailable paths
independently and renormalize their own timing weights, rather than dropping
the whole timing path from both components.

SA Placebo Timing Weights
-------------------------

SA placebo diagnostics also keep raw and standardized channels separate when
computing timing-weighted averages.  If a timing path has a raw placebo
estimate but no standardized estimate, the raw channel keeps that path and
renormalizes raw timing weights while the standardized channel drops only
the unavailable standardized path.

Dataset Registry
----------------

The root ``data/`` registry (see ``data/README.md``) documents all real-world
datasets available in the repository with provenance, cleaning rules, and
package-copy paths.
