diddesign
=========

Difference-in-differences (DID) remains the workhorse design for observational
causal inference in the social sciences. When multiple pre-treatment periods are
available, the standard two-period DID and sequential DID estimators exploit
different moment conditions, each valid under its own parallel-trends
assumption. **diddesign** implements the Egami and Yamauchi (2023, *Political
Analysis*) GMM framework that combines these component estimators with efficient
weighting, yielding the Double DID estimator.

The package provides a unified Python interface for standard DID, sequential
DID, Double DID, K-DID (three or more pre-treatment periods with adaptive
J-test moment selection), and staggered-adoption designs with lead-specific
estimates. All results are returned as immutable objects with pandas DataFrame
accessors for inspection, export, LaTeX table generation, and figure production.

Analysts working with panel data or repeated cross-sections will find a
complete workflow: pre-treatment placebo diagnostics via equivalence confidence
intervals, treatment effect estimation with bootstrap inference, GMM weight
inspection, and publication-ready output in a single package.

Getting started
---------------

Install with ``pip install diddesign``, then run the Quick Start example
in :doc:`user_guide` using the built-in Malesky (2014) dataset. Full
workflows are available in :doc:`tutorials`; concise snippets in
:doc:`examples`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user_guide
   tutorials
   examples
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
