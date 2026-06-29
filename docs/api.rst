API Reference
=============

The top-level ``diddesign`` namespace exposes all public functions and classes.
Import directly: ``from diddesign import did, did_check, DidResult``.

Core Estimation
---------------

.. autofunction:: diddesign.did

.. autofunction:: diddesign.did_check

Result Objects
--------------

.. autoclass:: diddesign.DidResult
   :members:

.. data:: diddesign.DIDResult

   Alias of :class:`diddesign.DidResult` retained for backward compatibility.

.. autoclass:: diddesign.DidCheckResult
   :members:

Estimate Row Types
~~~~~~~~~~~~~~~~~~

.. autoclass:: diddesign.DidEstimateRow
   :members:

.. autoclass:: diddesign.DidWeightRow
   :members:

.. autoclass:: diddesign.DidGmmRow
   :members:

.. autoclass:: diddesign.DidBootstrapDraw
   :members:

.. autoclass:: diddesign.DidBootstrapDrawK
   :members:

Diagnostic Row Types
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: diddesign.DidCheckDiagnosticRow
   :members:

.. autoclass:: diddesign.DidCheckTrendRow
   :members:

.. autoclass:: diddesign.DidCheckPatternRow
   :members:

Summary and Reporting
---------------------

.. autofunction:: diddesign.summary

.. autofunction:: diddesign.format_summary

.. autoclass:: diddesign.DiagnosticsReporter
   :members:

Formula Parsing
---------------

.. autofunction:: diddesign.did_formula

.. autoclass:: diddesign.DidFormulaSpec
   :members:

Plotting Rows
-------------

These functions prepare fitted result rows for downstream visualization or
export without rendering figures directly.

.. autofunction:: diddesign.fit

.. autofunction:: diddesign.check

Visualization
-------------

These functions render figures from fitted results. They require the optional
``diddesign[plot]`` dependency (matplotlib).

.. autofunction:: diddesign.plot_estimates

.. autofunction:: diddesign.plot_diagnostics

.. autofunction:: diddesign.plot_trends

.. autofunction:: diddesign.plot_placebo

.. autofunction:: diddesign.plot_pattern

Data Loading
------------

Built-in example datasets for quick-start workflows and reproducible examples.

.. automodule:: diddesign.data
   :members:
   :undoc-members:

Errors and Validation
---------------------

.. autoclass:: diddesign.DidDataError
   :members:
   :show-inheritance:

.. autoclass:: diddesign.DidError
   :members:
   :show-inheritance:

.. autoclass:: diddesign.DidValueError
   :members:
   :show-inheritance:

.. autoclass:: diddesign.DidRuntimeError
   :members:
   :show-inheritance:

.. autoclass:: diddesign.DidWarning
   :members:
   :show-inheritance:

.. autoclass:: diddesign.DataContractError
   :members:
   :show-inheritance:

.. autoclass:: diddesign.ErrorCode
   :members:
   :show-inheritance:

.. autoclass:: diddesign.WarningCode
   :members:
   :show-inheritance:

.. autofunction:: diddesign.did_warn

Package Metadata
----------------

.. data:: diddesign.__version__

   The installed package version string.
