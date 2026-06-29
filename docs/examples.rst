Examples
========

.. seealso::

   :doc:`tutorials` for narrative step-by-step workflows.
   :doc:`user_guide` for installation and parameter guidance.

Each example below demonstrates a specific scenario using the built-in
datasets. All outputs are produced with fixed random seeds for reproducibility.

Built-in Datasets
-----------------

The package ships two example datasets for immediate use:

.. code-block:: python

   from diddesign.data import data

   # Malesky et al. (2014): Vietnam communes, repeated cross-section
   malesky = data("malesky2014")
   print(malesky.shape)   # (6269, 42)

   # Paglayan (2019): US states teacher bargaining, staggered adoption panel
   paglayan = data("paglayan2019")
   print(paglayan.shape)  # (2058, 9)

RCS Double DID (Malesky 2014)
-----------------------------

A repeated cross-section workflow: estimate Double DID, run a lag-1 placebo
diagnostic, and inspect the component estimates and GMM weights.

.. code-block:: python

   from diddesign.data import data
   from diddesign import did, did_check, fit

   df = data("malesky2014")

   # Pre-treatment diagnostic
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

   # Treatment effect estimation
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

.. code-block:: python

   print(result.to_weights_frame())

.. code-block:: text

    lead    w_did    w_sdid  double_did_available
       0 1.806658 -0.806658                  True

.. code-block:: python

   # Plotting rows with diagnostic overlay
   fit_rows = fit(result, check_fit=check, as_frame=True)
   print(fit_rows)

Staggered Adoption (Paglayan 2019)
-----------------------------------

A panel staggered-adoption workflow: diagnose multiple pre-treatment lags,
then estimate lead-specific SA-Double-DID effects.

.. code-block:: python

   import numpy as np
   from diddesign.data import data
   from diddesign import did, did_check, fit

   df = data("paglayan2019")
   df["log_expenditure"] = np.log(df["pupil_expenditure"] + 1.0)

   # Multi-lag pre-treatment diagnostics
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

.. code-block:: python

   print(result.to_weights_frame())

.. code-block:: text

    lead    w_did   w_sdid  double_did_available
       0 0.843723 0.156277                  True

.. code-block:: python

   # Plotting rows (one per lead)
   fit_rows = fit(result, as_frame=True)
   print(fit_rows)

.. code-block:: text

   source     estimator  lag  lead  time_to_treat  estimate  std_error   ci90_lb  ci90_ub
      fit SA-Double-DID None     0              0  0.011401   0.012157 -0.008596 0.031398

K-DID with J-test (Paglayan 2019)
----------------------------------

When three or more pre-treatment periods are available, K-DID exploits
higher-order moment conditions. The J-test adaptively removes components
whose identifying assumptions appear violated:

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

The SA-kDID-3 component exploits three pre-treatment differences. Its larger
standard error reflects the additional noise from earlier pre-treatment periods.
The SA-K-DID combined estimate aggregates all valid components.

RCS with Covariates
-------------------

Adding continuous covariates to the Malesky RCS analysis:

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

Formula Interface
-----------------

The formula interface provides a concise alternative to specifying column names
individually. Covariates appear after the ``|`` separator, joined by ``+``:

.. code-block:: python

   from diddesign.data import data
   from diddesign import did

   df = data("malesky2014")
   result = did(
       df, formula="pro4 ~ treatment + post_treat | lnarea + lnpopden",
       time="year", data_type="rcs",
       id_cluster="id_district",
       n_boot=50, random_seed=1234,
   )
   print(result.to_dataframe())

.. code-block:: text

    estimator  lead  estimate  std_error     ci_lo    ci_hi    weight
   Double-DID     0  0.083374   0.042437  0.000199 0.166549       NaN
          DID     0  0.086286   0.054630 -0.020786 0.193359  1.864198
         sDID     0  0.089656   0.085487 -0.077895 0.257207 -0.864198

For categorical covariates, use the ``factor()`` wrapper:

.. code-block:: python

   result = did(
       df, formula="pro4 ~ treatment + post_treat | lnarea + factor(city)",
       time="year", data_type="rcs",
       id_cluster="id_district",
       n_boot=50, random_seed=1234,
   )

Visualization
-------------

Render figures from fitted results (requires ``diddesign[plot]``):

.. code-block:: python

   from diddesign import plot_estimates, plot_diagnostics, plot_trends

   # Event-study plot with placebo overlay
   plot_estimates(result, check_fit=check,
                  title="Double DID Estimates", save="estimates.png", show=False)

   # Multi-panel diagnostic figure
   plot_diagnostics(check, result=result,
                    title="Pre-treatment Diagnostics", save="diagnostics.png",
                    show=False)

   # Pre-treatment trend comparison
   plot_trends(check, title="Trend Comparison", save="trends.png", show=False)

LaTeX Table Export
------------------

Generate publication-ready tables directly from result objects:

.. code-block:: python

   from diddesign.data import data
   from diddesign import did

   df = data("malesky2014")
   result = did(
       df, outcome="pro4", treatment="treatment",
       time="year", post="post_treat", data_type="rcs",
       id_cluster="id_district", n_boot=200, random_seed=1234,
   )
   print(result.to_latex(caption="Recentralization Effect"))

.. code-block:: text

   \begin{table}[htbp]
   \centering
   \caption{Recentralization Effect}
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

Running Repository Examples
---------------------------

The repository ships standalone example scripts (for repository contributors):

.. code-block:: bash

   python3 examples/malesky_rcs_workflow.py
   python3 examples/paglayan_sa_workflow.py

These scripts print compact JSON summaries containing component estimates,
GMM weights, diagnostic rows, and plotting rows.

Dataset Provenance
------------------

For full dataset provenance, cleaning rules, and package-copy paths, see the
root ``data/`` registry at ``data/README.md``.
