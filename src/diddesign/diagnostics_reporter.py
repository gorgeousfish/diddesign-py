"""Automatic diagnostic report for DIDdesign estimation results.

Produces Stata-style diagnostic output including panel summary,
SA-specific diagnostics, and estimation result tables.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd


class DiagnosticsReporter:
    """Generate structured diagnostic reports for DidResult objects.

    Parameters
    ----------
    result : DidResult
        A completed estimation result object.
    verbose : int, default 1
        Output detail level:
        - 0: silent (no output)
        - 1: summary (panel stats + result table)
        - 2: detailed (+ GMM diagnostics, bootstrap stats, SA cohort detail)

    Examples
    --------
    >>> result = did(data, ...)
    >>> reporter = DiagnosticsReporter(result, verbose=2)
    >>> reporter.print_report()

    >>> # Or get as dict for programmatic use
    >>> info = reporter.to_dict()
    """

    _LINE_WIDTH = 60

    def __init__(self, result: Any, verbose: int = 1):
        self.result = result
        self.verbose = verbose
        self._warnings: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print full diagnostic report to stdout."""
        if self.verbose <= 0:
            return
        self._print_header()
        self._print_sample_info()
        if self._is_sa_design():
            self._print_sa_diagnostics()
        self._print_estimation_summary()
        if self.verbose >= 2:
            self._print_gmm_diagnostics()
            self._print_bootstrap_stats()
        self._print_footer()

    def to_dict(self) -> Dict[str, Any]:
        """Export diagnostic information as a dictionary."""
        info: Dict[str, Any] = {}
        info["design"] = self._design_label()
        info["n_observations"] = self._get_meta("n_obs")
        info["n_units"] = self._get_meta("n_units")
        info["n_periods"] = self._get_meta("n_periods")
        info["n_boot_requested"] = self._get_meta("n_boot_requested") or self._get_meta("n_boot")
        info["n_boot_realized"] = self._get_meta("n_boot_realized")

        # Estimates
        estimates_data: List[Dict[str, Any]] = []
        for row in self.result.estimates:
            estimates_data.append(row.as_dict())
        info["estimates"] = estimates_data

        # SA diagnostics
        if self._is_sa_design():
            info["sa_diagnostics"] = self._sa_diagnostics_dict()

        # GMM diagnostics
        gmm_rows = self.result.gmm_rows()
        if gmm_rows:
            info["gmm_diagnostics"] = self._gmm_diagnostics_dict(gmm_rows)

        # Bootstrap stats
        boot_info = self._bootstrap_stats_dict()
        if boot_info:
            info["bootstrap"] = boot_info

        # Warnings
        if self._warnings:
            info["warnings"] = list(self._warnings)

        return info

    def summary_frame(self) -> pd.DataFrame:
        """Return sample statistics as a DataFrame."""
        rows = []
        rows.append({"statistic": "design", "value": self._design_label()})
        n_obs = self._get_meta("n_obs")
        if n_obs is not None:
            rows.append({"statistic": "n_observations", "value": n_obs})
        n_units = self._get_meta("n_units")
        if n_units is not None:
            rows.append({"statistic": "n_units", "value": n_units})
        n_periods = self._get_meta("n_periods")
        if n_periods is not None:
            rows.append({"statistic": "n_periods", "value": n_periods})
        n_boot = self._get_meta("n_boot_requested") or self._get_meta("n_boot")
        if n_boot is not None:
            rows.append({"statistic": "n_boot_requested", "value": n_boot})
        n_boot_realized = self._get_meta("n_boot_realized")
        if n_boot_realized is not None:
            rows.append({"statistic": "n_boot_realized", "value": n_boot_realized})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_meta(self, key: str, default: Any = None) -> Any:
        """Safely retrieve metadata value."""
        try:
            return self.result.metadata.get(key, default)
        except (AttributeError, TypeError):
            return default

    def _is_sa_design(self) -> bool:
        """Check if this is a Staggered Adoption design."""
        design = self._get_meta("design")
        if design == "sa":
            return True
        # Also check estimator labels
        return any(
            row.estimator.startswith("SA-") for row in self.result.estimates
        )

    def _design_label(self) -> str:
        """Determine human-readable design label."""
        design = self._get_meta("design")
        # Check for K-DID
        has_kdid = any(
            row.estimator in {"K-DID", "SA-K-DID"}
            or row.estimator.startswith("kDID-")
            or row.estimator.startswith("SA-kDID-")
            for row in self.result.estimates
        )
        if design == "sa":
            if has_kdid:
                return "Staggered Adoption (SA-K-DID)"
            return "Staggered Adoption (SA)"
        if has_kdid:
            return "Standard (K-DID)"
        return "Standard DID"

    def _format_number(self, value: Any, fmt: str = ",.0f") -> str:
        """Format a number or return 'N/A' if unavailable."""
        if value is None:
            return "N/A"
        try:
            return format(value, fmt)
        except (TypeError, ValueError):
            return str(value)

    # ------------------------------------------------------------------
    # Print sections
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        """Print report header."""
        print()
        print("Double Difference-in-Differences Estimation")
        print("=" * self._LINE_WIDTH)
        print(f"{'Design:':<18}{self._design_label()}")

    def _print_sample_info(self) -> None:
        """Print panel summary statistics."""
        n_obs = self._get_meta("n_obs")
        n_units = self._get_meta("n_units")
        n_periods = self._get_meta("n_periods")

        if n_obs is not None:
            print(f"{'Observations:':<18}{self._format_number(n_obs)}")
        if n_units is not None:
            print(f"{'Units:':<18}{self._format_number(n_units)}")
        if n_periods is not None:
            print(f"{'Time periods:':<18}{self._format_number(n_periods)}")

        # Bootstrap info
        n_boot = self._get_meta("n_boot_requested") or self._get_meta("n_boot")
        if n_boot is not None:
            print(f"{'Bootstrap:':<18}{self._format_number(n_boot)} replications")

        # Confidence level
        level = self._get_meta("ci_level") or self._get_meta("level")
        if level is not None:
            try:
                pct = int(float(level) * 100) if float(level) <= 1.0 else int(level)
            except (TypeError, ValueError):
                pct = level
            print(f"{'Confidence:':<18}{pct}%")
        print()

    def _print_sa_diagnostics(self) -> None:
        """Print SA-specific diagnostic information."""
        print("SA Design Diagnostics")
        print("-" * self._LINE_WIDTH)

        # Identified and requested leads
        identified_leads = self._get_meta("identified_leads") or self._get_meta("identified_lead")
        requested_leads = self._get_meta("requested_leads") or self._get_meta("requested_lead")
        filtered_leads = self._get_meta("filtered_leads") or self._get_meta("filtered_lead")

        if identified_leads is not None and requested_leads is not None:
            n_identified = len(identified_leads)
            n_requested = len(requested_leads)
            thres = self._get_meta("thres")
            thres_str = f" (threshold = {thres})" if thres is not None else ""
            print(f"{'Valid periods:':<18}{n_identified} of {n_requested}{thres_str}")

        if identified_leads is not None:
            leads_str = ", ".join(str(l) for l in sorted(identified_leads))
            print(f"{'Leads estimated:':<18}[{leads_str}]")

        if filtered_leads:
            filtered_str = ", ".join(str(l) for l in sorted(filtered_leads))
            print(f"{'Leads filtered:':<18}[{filtered_str}]")

        # Cohort summary from metadata
        n_treated = self._get_meta("n_treated")
        n_control = self._get_meta("n_control")
        if n_treated is not None or n_control is not None:
            print()
            print("Cohort summary:")
            n_units = self._get_meta("n_units")
            if n_units is not None:
                print(f"  {'Total units:':<20}{self._format_number(n_units)}")
            if n_treated is not None:
                print(f"  {'Treated units:':<20}{self._format_number(n_treated)}")
            if n_control is not None:
                print(f"  {'Control units:':<20}{self._format_number(n_control)}")

        # Adoption-time distribution
        adoption_dist = self._get_meta("adoption_distribution")
        if adoption_dist is not None and isinstance(adoption_dist, dict):
            print()
            print("Adoption-time distribution:")
            total_adopted = sum(adoption_dist.values()) or 1
            for period, count in sorted(adoption_dist.items(), key=lambda x: x[0]):
                pct = 100.0 * count / total_adopted
                print(f"  Period {period}:{count:>6} units ({pct:.1f}%)")

        print()

    def _print_estimation_summary(self) -> None:
        """Print the estimation result table."""
        print("Results")
        print("-" * self._LINE_WIDTH)

        estimates = self.result.estimates
        if not estimates:
            print("  (no estimates available)")
            print()
            return

        # Header
        header = f"{'':18}{'Estimate':>10}{'Std.Err':>10}{'CI Low':>10}{'CI High':>10}{'Weight':>8}"
        print(header)

        # Separate components from combined estimates
        component_rows = []
        combined_rows = []
        for row in estimates:
            if row.estimator.endswith("Double-DID") or row.estimator in {"K-DID", "SA-K-DID"}:
                combined_rows.append(row)
            else:
                component_rows.append(row)

        # Print component rows
        for row in component_rows:
            self._print_estimate_row(row)

        # Separator before combined
        if combined_rows and component_rows:
            print("\u2500" * self._LINE_WIDTH)

        # Print combined rows
        for row in combined_rows:
            self._print_estimate_row(row)

        print()

    def _print_estimate_row(self, row: Any) -> None:
        """Print a single estimate row."""
        label = self._estimator_display_label(row.estimator)
        estimate_str = f"{row.estimate:>10.4f}"
        se_str = f"{row.std_error:>10.4f}" if row.std_error is not None else f"{'N/A':>10}"
        ci_lo_str = f"{row.ci_lo:>10.4f}" if row.ci_lo is not None else f"{'N/A':>10}"
        ci_hi_str = f"{row.ci_hi:>10.4f}" if row.ci_hi is not None else f"{'N/A':>10}"
        weight_str = f"{row.weight:>8.3f}" if row.weight is not None else f"{'.':>8}"
        print(f"{label:<18}{estimate_str}{se_str}{ci_lo_str}{ci_hi_str}{weight_str}")

    def _estimator_display_label(self, estimator: str) -> str:
        """Map internal estimator label to display label."""
        display_map = {
            "DID": "DID",
            "sDID": "Sequential DID",
            "Double-DID": "Double DID",
            "SA-DID": "SA-DID",
            "SA-sDID": "SA-Sequential DID",
            "SA-Double-DID": "SA-Double DID",
            "K-DID": "K-DID",
            "SA-K-DID": "SA-K-DID",
        }
        return display_map.get(estimator, estimator)

    def _print_gmm_diagnostics(self) -> None:
        """Print GMM weight diagnostics (verbose >= 2)."""
        gmm_rows = self.result.gmm_rows()
        if not gmm_rows:
            # Try K-DID weights
            k_weights = self._get_meta("k_weights_by_lead")
            if k_weights:
                self._print_k_gmm_diagnostics(k_weights)
            return

        print("GMM Weight Diagnostics")
        print("-" * self._LINE_WIDTH)

        for gmm_row in gmm_rows:
            if len(gmm_rows) > 1:
                print(f"  Lead {gmm_row.lead}:")
                prefix = "    "
            else:
                prefix = ""

            # VCOV condition number
            if gmm_row.vcov_did is not None and gmm_row.vcov_sdid is not None:
                cond = self._compute_vcov_condition(
                    gmm_row.vcov_did, gmm_row.vcov_sdid, gmm_row.vcov_covariance
                )
                if cond is not None:
                    print(f"{prefix}{'VCOV condition number:':<24}{cond:.1f}")
                    if cond > 1000:
                        self._warnings.append(
                            f"Lead {gmm_row.lead}: VCOV matrix is near-singular (cond={cond:.1f})"
                        )

            # Weight sum check
            if gmm_row.w_did is not None and gmm_row.w_sdid is not None:
                weight_sum = gmm_row.w_did + gmm_row.w_sdid
                print(f"{prefix}{'Weight sum check:':<24}{weight_sum:.4f}")

                # Weight range
                w_min = min(gmm_row.w_did, gmm_row.w_sdid)
                w_max = max(gmm_row.w_did, gmm_row.w_sdid)
                in_range = "within [0,1]" if 0 <= w_min and w_max <= 1 else "OUTSIDE [0,1]"
                print(f"{prefix}{'Weight range:':<24}[{w_min:.3f}, {w_max:.3f}] ({in_range})")

                if w_min < 0 or w_max > 1:
                    self._warnings.append(
                        f"Lead {gmm_row.lead}: GMM weights outside [0,1] range "
                        f"(w_did={gmm_row.w_did:.4f}, w_sdid={gmm_row.w_sdid:.4f})"
                    )

            # GMM variance
            if gmm_row.gmm_variance is not None:
                print(f"{prefix}{'GMM variance:':<24}{gmm_row.gmm_variance:.6f}")

        # J-test info if available
        jtest_stat = self._get_meta("jtest_stat")
        jtest_df = self._get_meta("jtest_df")
        jtest_pval = self._get_meta("jtest_pval")
        if jtest_stat is not None:
            df_str = f"df={jtest_df}" if jtest_df is not None else ""
            p_str = f", p={jtest_pval:.3f}" if jtest_pval is not None else ""
            print(f"{'J-test statistic:':<24}{jtest_stat:.2f} ({df_str}{p_str})")

        print()

    def _print_k_gmm_diagnostics(self, k_weights: Any) -> None:
        """Print K-DID GMM diagnostics."""
        print("K-DID Weight Diagnostics")
        print("-" * self._LINE_WIDTH)

        for lead, info in sorted(k_weights.items(), key=lambda x: x[0]):
            w_k = info.get("w_k", ())
            jtest_stat = info.get("jtest_stat")
            jtest_df = info.get("jtest_df")
            jtest_pval = info.get("jtest_pval")
            dropped = info.get("dropped_moments", ())

            print(f"  Lead {lead}:")
            if w_k:
                weights_str = ", ".join(f"{w:.4f}" for w in w_k)
                print(f"    Weights: [{weights_str}]")
                w_sum = sum(w_k)
                print(f"    Weight sum: {w_sum:.4f}")
            if jtest_stat is not None:
                df_str = f"df={jtest_df}" if jtest_df is not None else ""
                p_str = f", p={jtest_pval:.3f}" if jtest_pval is not None else ""
                print(f"    J-test: {jtest_stat:.2f} ({df_str}{p_str})")
            if dropped:
                print(f"    Dropped moments: {list(dropped)}")

        print()

    def _print_bootstrap_stats(self) -> None:
        """Print bootstrap summary statistics (verbose >= 2)."""
        n_boot_requested = self._get_meta("n_boot_requested") or self._get_meta("n_boot")
        n_boot_realized = self._get_meta("n_boot_realized")

        if n_boot_requested is None and n_boot_realized is None:
            return

        print("Bootstrap Summary")
        print("-" * self._LINE_WIDTH)

        if n_boot_requested is not None:
            print(f"{'Total iterations:':<22}{self._format_number(n_boot_requested)}")

        if n_boot_realized is not None:
            if n_boot_requested is not None and n_boot_requested > 0:
                pct = 100.0 * n_boot_realized / n_boot_requested
                print(f"{'Successful:':<22}{self._format_number(n_boot_realized)} ({pct:.1f}%)")
                n_failed = n_boot_requested - n_boot_realized
                if n_failed > 0:
                    fail_pct = 100.0 * n_failed / n_boot_requested
                    print(f"{'Failed:':<22}{self._format_number(n_failed)} ({fail_pct:.1f}%)")
                    self._warnings.append(
                        f"{n_failed} of {n_boot_requested} bootstrap iterations failed"
                    )
            else:
                print(f"{'Successful:':<22}{self._format_number(n_boot_realized)}")

        print()

    def _print_footer(self) -> None:
        """Print footer with warnings if any."""
        if self._warnings:
            print("Warnings")
            print("-" * self._LINE_WIDTH)
            for w in self._warnings:
                print(f"  * {w}")
            print()

    # ------------------------------------------------------------------
    # Helper computations
    # ------------------------------------------------------------------

    def _compute_vcov_condition(
        self, vcov_did: float, vcov_sdid: float, vcov_cov: Optional[float]
    ) -> Optional[float]:
        """Compute VCOV matrix condition number."""
        if vcov_cov is None:
            vcov_cov = 0.0
        # 2x2 matrix eigenvalues
        trace = vcov_did + vcov_sdid
        det = vcov_did * vcov_sdid - vcov_cov * vcov_cov
        discriminant = trace * trace - 4.0 * det
        if discriminant < 0:
            return None
        sqrt_disc = math.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2.0
        lambda2 = (trace - sqrt_disc) / 2.0
        if lambda2 <= 0 or lambda1 <= 0:
            return None
        return lambda1 / lambda2

    def _sa_diagnostics_dict(self) -> Dict[str, Any]:
        """Gather SA diagnostics as a dictionary."""
        info: Dict[str, Any] = {}
        identified_leads = self._get_meta("identified_leads") or self._get_meta("identified_lead")
        requested_leads = self._get_meta("requested_leads") or self._get_meta("requested_lead")
        filtered_leads = self._get_meta("filtered_leads") or self._get_meta("filtered_lead")

        if identified_leads is not None:
            info["identified_leads"] = list(identified_leads)
        if requested_leads is not None:
            info["requested_leads"] = list(requested_leads)
        if filtered_leads is not None:
            info["filtered_leads"] = list(filtered_leads)
        info["thres"] = self._get_meta("thres")
        info["n_treated"] = self._get_meta("n_treated")
        info["n_control"] = self._get_meta("n_control")
        info["adoption_distribution"] = self._get_meta("adoption_distribution")
        return info

    def _gmm_diagnostics_dict(self, gmm_rows: Any) -> Dict[str, Any]:
        """Gather GMM diagnostics as a dictionary."""
        info: Dict[str, Any] = {"leads": []}
        for gmm_row in gmm_rows:
            lead_info: Dict[str, Any] = {"lead": gmm_row.lead}
            if gmm_row.w_did is not None:
                lead_info["w_did"] = gmm_row.w_did
            if gmm_row.w_sdid is not None:
                lead_info["w_sdid"] = gmm_row.w_sdid
            if gmm_row.gmm_variance is not None:
                lead_info["gmm_variance"] = gmm_row.gmm_variance
            if gmm_row.vcov_did is not None:
                lead_info["vcov_condition"] = self._compute_vcov_condition(
                    gmm_row.vcov_did, gmm_row.vcov_sdid, gmm_row.vcov_covariance
                )
            info["leads"].append(lead_info)
        return info

    def _bootstrap_stats_dict(self) -> Optional[Dict[str, Any]]:
        """Gather bootstrap statistics as a dictionary."""
        n_boot_requested = self._get_meta("n_boot_requested") or self._get_meta("n_boot")
        n_boot_realized = self._get_meta("n_boot_realized")
        if n_boot_requested is None and n_boot_realized is None:
            return None
        info: Dict[str, Any] = {}
        if n_boot_requested is not None:
            info["n_requested"] = n_boot_requested
        if n_boot_realized is not None:
            info["n_realized"] = n_boot_realized
            if n_boot_requested is not None:
                info["n_failed"] = n_boot_requested - n_boot_realized
                info["success_rate"] = (
                    n_boot_realized / n_boot_requested if n_boot_requested > 0 else None
                )
        return info


__all__ = ["DiagnosticsReporter"]
