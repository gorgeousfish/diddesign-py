Tutorials
=========

.. seealso::

   :doc:`examples` for concise copy-paste snippets.
   :doc:`api` for the complete API reference.

These tutorials walk through complete workflows from data loading to
publication-ready output. Each uses the built-in datasets and produces
real output that can be verified by running the code blocks sequentially.

Minimal RCS Double DID
----------------------

A complete Double DID workflow on a small repeated cross-section dataset
demonstrates the estimation mechanics:

.. code-block:: python

   import pandas as pd
   from diddesign import did, summary

   df = pd.DataFrame(
       {
           "cluster": [
               "north-pre0-control",
               "north-pre0-treated",
               "south-pre0-control",
               "south-pre0-treated",
               "north-pre1-control",
               "north-pre1-treated",
               "south-pre1-control",
               "south-pre1-treated",
               "north-post-control",
               "north-post-treated",
               "south-post-control",
               "south-post-treated",
           ],
           "time": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
           "post": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
           "treated": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
           "y": [1.0, 2.0, 1.2, 2.2, 2.0, 3.0, 2.2, 3.2, 3.0, 5.5, 3.2, 5.7],
       }
   )

   result = did(
       df,
       outcome="y",
       treatment="treated",
       time="time",
       post="post",
       id_cluster="cluster",
       data_type="rcs",
       n_boot=30,
       random_seed=1234,
   )

   print(summary(result, as_frame=True))

The result contains three rows: Double-DID (GMM-optimal combination), DID,
and sDID (sequential DID). The Double-DID estimate is a weighted combination
of DID and sDID where the weights minimize asymptotic variance under the
efficient-GMM criterion.

Inspecting Component Weights
----------------------------

The Double-DID result decomposes into component estimators and their GMM
weights. Read this decomposition before reporting the combined row:

.. code-block:: python

   estimates = result.to_estimates_frame()
   weights = result.to_weights_frame()
   gmm_rows = result.to_gmm_frame()

   # Separate component and combined rows
   component_rows = estimates.loc[estimates["estimator"].isin(["DID", "sDID"])]
   combined_row = estimates.loc[estimates["estimator"] == "Double-DID"].iloc[0]
   weight_row = weights.iloc[0]

   # Verify the GMM recomposition
   did_component = component_rows.loc[
       component_rows["estimator"] == "DID", "estimate"
   ].iloc[0]
   sdid_component = component_rows.loc[
       component_rows["estimator"] == "sDID", "estimate"
   ].iloc[0]

   recomposed = (
       weight_row["w_did"] * did_component
       + weight_row["w_sdid"] * sdid_component
   )

   print(component_rows[["estimator", "estimate", "std_error", "weight"]])
   print(f"w_did={weight_row['w_did']:.4f}, w_sdid={weight_row['w_sdid']:.4f}")
   print(f"Double-DID estimate: {combined_row['estimate']:.6f}")
   print(f"Recomposed from components: {recomposed:.6f}")
   print(gmm_rows[["vcov_did", "vcov_sdid", "vcov_covariance", "gmm_variance"]])

If either component weight is negative or exceeds one, the combined row
represents an extrapolating GMM recomposition rather than a simple convex
average. This occurs when the covariance between DID and sDID is large enough
that the variance-minimizing combination places negative weight on one
component.

Real-Data RCS Workflow (Malesky 2014)
-------------------------------------

The Malesky 2014 dataset studies administrative recentralization in Vietnam
using repeated cross-sections of commune-level public goods outcomes.

.. code-block:: python

   from diddesign.data import data
   from diddesign import did, did_check, summary

   df = data("malesky2014")

   # Pre-treatment placebo diagnostic
   check = did_check(
       data=df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district", lag=[1], n_boot=50, random_seed=1234,
   )
   print(check.to_summary_frame())

.. code-block:: text

    lag  estimate_raw  std_error_raw  eqci95_lb_std  eqci95_ub_std
      1      -0.00337       0.041026      -0.163403       0.163403

The placebo estimate of -0.003 is economically negligible and the 95%
equivalence confidence interval is symmetric around zero, consistent with
parallel trends in the pre-treatment period.

.. code-block:: python

   # Estimate treatment effects
   result = did(
       df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district", n_boot=200, random_seed=1234,
   )
   print(result.to_dataframe())

.. code-block:: text

    estimator  lead  estimate  std_error     ci_lo     ci_hi    weight
   Double-DID     0  0.076596   0.046146 -0.013849  0.167041       NaN
          DID     0  0.079314   0.057338 -0.033066  0.191694  1.806658
         sDID     0  0.082684   0.089100 -0.091949  0.257317 -0.806658

The DID weight exceeds 1 and sDID weight is negative. This means the
GMM-optimal combination extrapolates: it places more weight on DID (which has
lower variance) and negative weight on sDID. The combined Double-DID standard
error (0.046) is lower than either component alone.

.. code-block:: python

   print(result.to_weights_frame())

.. code-block:: text

    lead    w_did    w_sdid  double_did_available
       0 1.806658 -0.806658                  True

Staggered Adoption Workflow (Paglayan 2019)
-------------------------------------------

The Paglayan 2019 dataset is a balanced panel of 49 US states from 1959 to
2000, studying the effects of collective bargaining laws on education spending.

.. code-block:: python

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

.. code-block:: text

    lag  estimate_raw  std_error_raw  eqci95_lb_std  eqci95_ub_std
      1     -0.002669       0.009736      -0.117499       0.117499
      2     -0.012447       0.007841      -0.151357       0.151357
      3      0.002269       0.011331      -0.121691       0.121691

All three pre-treatment lag estimates are close to zero with symmetric
equivalence intervals, supporting the parallel trends assumption at each lag.

.. code-block:: python

   # SA-Double-DID estimation
   result = did(
       df, outcome="log_expenditure", treatment="treatment",
       time="year", unit_id="state", design="sa",
       thres=1, n_boot=200, random_seed=1234,
   )
   print(result.to_dataframe())

.. code-block:: text

       estimator  lead  estimate  std_error     ci_lo     ci_hi    weight
   SA-Double-DID     0  0.011401   0.012157 -0.011430  0.033800       NaN
          SA-DID     0  0.010984   0.012247 -0.011420  0.034097  0.843723
         SA-sDID     0  0.013653   0.014537 -0.014634  0.042717  0.156277

In this staggered-adoption design, the SA-Double-DID estimate combines SA-DID
and SA-sDID with weights 0.84 and 0.16 respectively. Both weights are positive,
indicating a standard convex combination where SA-DID contributes most of the
information.

K-DID with J-test
-----------------

When the panel has three or more pre-treatment periods, K-DID exploits
higher-order moment conditions. The J-test adaptively removes components whose
identifying assumptions appear violated:

.. code-block:: python

   result_k = did(
       df, outcome="log_expenditure", treatment="treatment",
       time="year", unit_id="state", design="sa",
       kmax=3, jtest=True, thres=1,
       n_boot=200, random_seed=1234,
   )
   print(result_k.to_dataframe())

.. code-block:: text

   estimator  lead  estimate  std_error     ci_lo    ci_hi weight
    SA-K-DID     0  0.011685   0.012156 -0.011180 0.034105   None
      SA-DID     0  0.010984   0.012247 -0.011420 0.034097   None
     SA-sDID     0  0.013653   0.014537 -0.014634 0.042717   None
   SA-kDID-3     0  0.003875   0.023613 -0.040192 0.052995   None

The SA-K-DID row is the combined estimator exploiting all valid moment
conditions. The SA-kDID-3 row is the third-order component based on t-3
pre-treatment differences. Its larger standard error reflects the additional
noise from using earlier pre-treatment periods.

Preparing Plotting Rows
------------------------

``fit()`` prepares event-time plotting records as a pandas DataFrame without
drawing a figure directly:

.. code-block:: python

   from diddesign import fit

   fit_rows = fit(result, as_frame=True)
   print(fit_rows)

.. code-block:: text

   source     estimator  lag  lead  time_to_treat  estimate  std_error   ci90_lb  ci90_ub
      fit SA-Double-DID None     0              0  0.011401   0.012157 -0.008596 0.031398

Each row contains the lead, estimate, standard error, and 90% confidence
interval endpoints (``ci90_lb``, ``ci90_ub``) that downstream plotting code
renders or exports. When a ``check_fit`` overlay is supplied, diagnostic
rows are appended with appropriate labels:

.. code-block:: python

   fit_rows_with_check = fit(result, check_fit=check, as_frame=True)
   print(fit_rows_with_check)

The native ``plot_*`` helpers consume these same rows:

.. code-block:: python

   from diddesign import plot_estimates

   plot_estimates(result, title="SA-Double-DID", save="sa_estimates.png", show=False)

LaTeX Export
------------

Generate a publication-ready LaTeX table from fitted results:

.. code-block:: python

   from diddesign.data import data
   from diddesign import did

   df = data("malesky2014")
   result = did(
       df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district", n_boot=200, random_seed=1234,
   )
   latex_str = result.to_latex(caption="Recentralization Effect on Pro4")
   print(latex_str)

.. code-block:: text

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

The LaTeX output includes significance stars and standard formatting suitable
for direct inclusion in manuscripts.

Using Covariates
----------------

Covariates enter the estimation model as additional regressors. The following
workflow adds commune area and population density controls to the Malesky
analysis:

.. code-block:: python

   from diddesign.data import data
   from diddesign import did

   df = data("malesky2014")
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

Comparing with the baseline (no covariates) estimate of 0.077, adding
covariates shifts the point estimate slightly upward to 0.083 and narrows
the confidence interval, with the lower bound now crossing zero.

The same analysis can use the formula interface:

.. code-block:: python

   result_formula = did(
       df, formula="pro4 ~ treatment + post_treat | lnarea + lnpopden",
       time="year", data_type="rcs",
       id_cluster="id_district",
       n_boot=50, random_seed=1234,
   )
   print(result_formula.to_dataframe())

.. code-block:: text

    estimator  lead  estimate  std_error     ci_lo    ci_hi    weight
   Double-DID     0  0.083374   0.042437  0.000199 0.166549       NaN
          DID     0  0.086286   0.054630 -0.020786 0.193359  1.864198
         sDID     0  0.089656   0.085487 -0.077895 0.257207 -0.864198

Both approaches produce identical results.
