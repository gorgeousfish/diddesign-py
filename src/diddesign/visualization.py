"""Native matplotlib rendering for DIDdesign plotting data.

This module provides publication-grade figure rendering using the data rows
returned by :func:`diddesign.plotting.fit` and :func:`diddesign.plotting.check`.

Install the optional plotting dependency with::

    pip install diddesign[plot]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import DidCheckResult
from .plotting import check as _check_data, fit as _fit_data
from .results.objects import DidResult


def _require_matplotlib():
    """Import matplotlib at runtime; raise a helpful error if missing."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install diddesign[plot]"
        ) from None


def _apply_publication_style(ax, *, title: str | None, xlabel: str | None, ylabel: str | None) -> None:
    """Apply publication-grade styling to an Axes object."""
    import matplotlib as mpl

    # Show all four spines but make top/right thinner
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["top"].set_linewidth(0.4)
    ax.spines["right"].set_linewidth(0.4)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", direction="in", labelsize=8, width=0.8)
    ax.tick_params(axis="both", which="minor", direction="in", width=0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _apply_default_style(ax, *, title: str | None, xlabel: str | None, ylabel: str | None) -> None:
    """Apply default matplotlib styling."""
    if title:
        ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _finalize_figure(fig, ax, *, save: str | None, dpi: int, show: bool, style: str, _skip_layout: bool = False):
    """Tight layout, optional save, optional show."""
    _, plt = _require_matplotlib()
    if not _skip_layout:
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _get_or_create_axes(ax, figsize: tuple[float, float]):
    """Return (fig, ax, external) from an existing ax or create new."""
    _, plt = _require_matplotlib()
    if ax is not None:
        fig = ax.get_figure()
        return fig, ax, True
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, False


def _set_publication_font():
    """Try to set serif font for publication style."""
    import matplotlib as mpl

    serif_fonts = ["DejaVu Serif", "Times New Roman", "serif"]
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = serif_fonts
    mpl.rcParams["mathtext.fontset"] = "dejavuserif"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_estimates(
    result: DidResult,
    *,
    check_fit: DidCheckResult | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    style: str = "publication",
    ci_level: float = 0.90,
    save: str | None = None,
    dpi: int = 150,
    ax: Any | None = None,
    show: bool = True,
) -> Any:
    """Plot Double-DID and SA-Double-DID effect estimates with confidence intervals.

    Higher-order K-DID results remain available through the returned estimate,
    bootstrap, weight, and GMM frames. This helper renders the Double-DID-style
    display rows produced by :func:`diddesign.fit`.

    Parameters
    ----------
    result : DidResult
        The fitted DID result object.
    check_fit : DidCheckResult, optional
        If provided, placebo estimates are overlaid on the negative time axis.
    title : str, optional
        Custom figure title.
    xlabel : str, optional
        Custom x-axis label.
    ylabel : str, optional
        Custom y-axis label.
    figsize : tuple
        Figure dimensions in inches.
    style : str
        ``"publication"`` for serif/clean style; ``"default"`` for matplotlib defaults.
    ci_level : float
        Confidence level (default 0.90).
    save : str, optional
        If provided, save figure to this file path.
    dpi : int
        Resolution for saved figure.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw on.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, plt = _require_matplotlib()
    from statistics import NormalDist

    if style == "publication":
        _set_publication_font()

    # Get data using the existing plotting data layer
    plot_df = _fit_data(result, check_fit=check_fit, as_frame=True)

    # Recompute CI bounds if ci_level != 0.90
    z = NormalDist().inv_cdf(1 - (1 - ci_level) / 2)
    plot_df = plot_df.copy()
    plot_df["ci_lb"] = plot_df["estimate"] - z * plot_df["std_error"]
    plot_df["ci_ub"] = plot_df["estimate"] + z * plot_df["std_error"]

    fig, ax_, _external = _get_or_create_axes(ax, figsize)

    # Separate check (placebo) and fit rows
    check_rows = plot_df[plot_df["source"] == "check"]
    fit_rows = plot_df[plot_df["source"] == "fit"]

    # Plot placebo estimates (pre-treatment)
    if not check_rows.empty:
        ax_.errorbar(
            check_rows["time_to_treat"],
            check_rows["estimate"],
            yerr=[
                check_rows["estimate"] - check_rows["ci_lb"],
                check_rows["ci_ub"] - check_rows["estimate"],
            ],
            fmt="s",
            color="#666666",
            ecolor="#999999",
            capsize=3,
            markersize=5,
            label="Placebo",
            linewidth=1.2,
            elinewidth=1.0,
        )

    # Plot treatment estimates
    if not fit_rows.empty:
        fit_estimators = [str(value) for value in fit_rows["estimator"].dropna().unique()]
        fit_label = fit_estimators[0] if len(fit_estimators) == 1 else "Fitted estimates"
        ax_.errorbar(
            fit_rows["time_to_treat"],
            fit_rows["estimate"],
            yerr=[
                fit_rows["estimate"] - fit_rows["ci_lb"],
                fit_rows["ci_ub"] - fit_rows["estimate"],
            ],
            fmt="o",
            color="#000000",
            ecolor="#333333",
            capsize=3,
            markersize=6,
            label=fit_label,
            linewidth=1.2,
            elinewidth=1.0,
        )

    # Reference line at y = 0
    ax_.axhline(y=0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    # Vertical line at x = 0 (treatment onset)
    ax_.axvline(x=0, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)

    # Force integer X-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax_.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust axis range for sparse data
    all_x = pd.concat([check_rows["time_to_treat"], fit_rows["time_to_treat"]])
    if len(all_x) <= 2:
        ax_.set_xlim(all_x.min() - 0.8, all_x.max() + 0.8)

    # Labels and legend
    default_title = "Effect Estimates"
    default_xlabel = "Time to Treatment"
    default_ylabel = "Estimate"

    if style == "publication":
        _apply_publication_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )
    else:
        _apply_default_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )

    if not check_rows.empty or fit_rows.shape[0] > 1:
        ax_.legend(frameon=True, fontsize=9, edgecolor="#CCCCCC", fancybox=False)

    return _finalize_figure(fig, ax_, save=save, dpi=dpi, show=show, style=style, _skip_layout=_external)


def plot_trends(
    check_result: DidCheckResult,
    *,
    ci: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    style: str = "publication",
    ci_level: float = 0.90,
    save: str | None = None,
    dpi: int = 150,
    ax: Any | None = None,
    show: bool = True,
) -> Any:
    """Plot treated vs. control outcome mean trends.

    Parameters
    ----------
    check_result : DidCheckResult
        Diagnostic check result containing trend rows.
    ci : bool
        Whether to show confidence bands.
    title : str, optional
        Custom figure title.
    xlabel, ylabel : str, optional
        Custom axis labels.
    figsize : tuple
        Figure dimensions.
    style : str
        ``"publication"`` or ``"default"``.
    ci_level : float
        Confidence level for bands.
    save : str, optional
        File path to save figure.
    dpi : int
        Resolution for saved figure.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, plt = _require_matplotlib()
    from statistics import NormalDist

    if style == "publication":
        _set_publication_font()

    # Get trends data
    check_data = _check_data(check_result, as_frame=True)
    if "trends" not in check_data:
        raise ValueError("plot_trends requires trend data (non-SA design check result).")

    trends_df = check_data["trends"]

    fig, ax_, _external = _get_or_create_axes(ax, figsize)

    # Recompute CI if needed
    z = NormalDist().inv_cdf(1 - (1 - ci_level) / 2)

    # Separate treated and control groups (labels are capitalized: "Treated"/"Control")
    treated = trends_df[trends_df["group"] == "Treated"].sort_values("time_to_treat")
    control = trends_df[trends_df["group"] == "Control"].sort_values("time_to_treat")

    # Plot treated line
    ax_.plot(
        treated["time_to_treat"],
        treated["outcome_mean"],
        "o-",
        color="#000000",
        linewidth=1.5,
        markersize=5,
        label="Treated",
    )

    # Plot control line
    ax_.plot(
        control["time_to_treat"],
        control["outcome_mean"],
        "s--",
        color="#666666",
        linewidth=1.5,
        markersize=5,
        label="Control",
    )

    # Optional CI bands
    if ci:
        if not treated.empty:
            ci_lb = treated["outcome_mean"] - z * treated["outcome_sd"]
            ci_ub = treated["outcome_mean"] + z * treated["outcome_sd"]
            ax_.fill_between(
                treated["time_to_treat"],
                ci_lb,
                ci_ub,
                alpha=0.15,
                color="#000000",
            )
        if not control.empty:
            ci_lb = control["outcome_mean"] - z * control["outcome_sd"]
            ci_ub = control["outcome_mean"] + z * control["outcome_sd"]
            ax_.fill_between(
                control["time_to_treat"],
                ci_lb,
                ci_ub,
                alpha=0.15,
                color="#666666",
            )

    # Shade post-treatment region
    all_times = trends_df["time_to_treat"].values
    post_times = all_times[all_times >= 0]
    if len(post_times) > 0:
        x_max = post_times.max()
        ax_.axvspan(-0.5, x_max + 0.5, alpha=0.08, color="#AAAAAA", zorder=0,
                    label="Post-treatment")

    # Vertical line at t=0
    ax_.axvline(x=-0.5, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)

    # Force integer X-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax_.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels
    default_title = "Outcome Trends"
    default_xlabel = "Time to Treatment"
    default_ylabel = "Outcome Mean"

    if style == "publication":
        _apply_publication_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )
    else:
        _apply_default_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )

    ax_.legend(frameon=True, fontsize=9, edgecolor="#CCCCCC", fancybox=False)

    return _finalize_figure(fig, ax_, save=save, dpi=dpi, show=show, style=style, _skip_layout=_external)


def plot_placebo(
    check_result: DidCheckResult,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    style: str = "publication",
    ci_level: float = 0.90,
    save: str | None = None,
    dpi: int = 150,
    ax: Any | None = None,
    show: bool = True,
) -> Any:
    """Plot pre-treatment placebo test results.

    Parameters
    ----------
    check_result : DidCheckResult
        Diagnostic check result containing placebo rows.
    title : str, optional
        Custom figure title.
    xlabel, ylabel : str, optional
        Custom axis labels.
    figsize : tuple
        Figure dimensions.
    style : str
        ``"publication"`` or ``"default"``.
    ci_level : float
        Confidence level for error bars (default 0.90).
    save : str, optional
        File path to save figure.
    dpi : int
        Resolution for saved figure.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, plt = _require_matplotlib()
    from statistics import NormalDist

    if style == "publication":
        _set_publication_font()

    # Get placebo data
    check_data = _check_data(check_result, as_frame=True)
    placebo_df = check_data["placebo"]

    if placebo_df.empty:
        raise ValueError("plot_placebo requires non-empty placebo data.")

    fig, ax_, _external = _get_or_create_axes(ax, figsize)

    z = NormalDist().inv_cdf(1 - (1 - ci_level) / 2)

    # Compute CI for standardized estimates
    placebo_df = placebo_df.copy()
    placebo_df["ci_lb"] = placebo_df["estimate_std"] - z * placebo_df["std_error_std"]
    placebo_df["ci_ub"] = placebo_df["estimate_std"] + z * placebo_df["std_error_std"]

    # Plot equivalence CI band (from eqci bounds)
    if "eqci95_lb_std" in placebo_df.columns and "eqci95_ub_std" in placebo_df.columns:
        eq_lb = placebo_df["eqci95_lb_std"].min()
        eq_ub = placebo_df["eqci95_ub_std"].max()
        x_range = [placebo_df["time_to_treat"].min() - 0.5, placebo_df["time_to_treat"].max() + 0.5]
        ax_.fill_between(
            x_range,
            eq_lb,
            eq_ub,
            alpha=0.10,
            color="#4477AA",
            label="Equivalence Band",
            zorder=0,
        )

    # Plot standardized placebo estimates with error bars
    ax_.errorbar(
        placebo_df["time_to_treat"],
        placebo_df["estimate_std"],
        yerr=[
            placebo_df["estimate_std"] - placebo_df["ci_lb"],
            placebo_df["ci_ub"] - placebo_df["estimate_std"],
        ],
        fmt="o",
        color="#000000",
        ecolor="#333333",
        capsize=3,
        markersize=6,
        linewidth=1.2,
        elinewidth=1.0,
        label="Standardized Placebo",
    )

    # Reference line at y = 0
    ax_.axhline(y=0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    # Force integer X-axis ticks
    from matplotlib.ticker import MaxNLocator
    ax_.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust axis range for sparse data (single point)
    x_vals = placebo_df["time_to_treat"]
    if len(x_vals) <= 2:
        ax_.set_xlim(x_vals.min() - 0.8, x_vals.max() + 0.8)

    # Labels
    default_title = "Placebo Test"
    default_xlabel = "Time to Treatment"
    default_ylabel = "Standardized Estimate"

    if style == "publication":
        _apply_publication_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )
    else:
        _apply_default_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )

    ax_.legend(frameon=True, fontsize=9, edgecolor="#CCCCCC", fancybox=False)

    return _finalize_figure(fig, ax_, save=save, dpi=dpi, show=show, style=style, _skip_layout=_external)


def plot_pattern(
    check_result: DidCheckResult,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    style: str = "publication",
    save: str | None = None,
    dpi: int = 150,
    ax: Any | None = None,
    show: bool = True,
) -> Any:
    """Plot staggered-adoption treatment timing heatmap.

    Parameters
    ----------
    check_result : DidCheckResult
        Diagnostic check result containing pattern rows (SA design).
    title : str, optional
        Custom figure title.
    xlabel, ylabel : str, optional
        Custom axis labels.
    figsize : tuple
        Figure dimensions.
    style : str
        ``"publication"`` or ``"default"``.
    save : str, optional
        File path to save figure.
    dpi : int
        Resolution for saved figure.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, plt = _require_matplotlib()
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    if style == "publication":
        _set_publication_font()

    # Get pattern data
    check_data = _check_data(check_result, as_frame=True)
    if "pattern" not in check_data:
        raise ValueError("plot_pattern requires staggered-adoption (SA) design check result.")

    pattern_df = check_data["pattern"]

    fig, ax_, _external = _get_or_create_axes(ax, figsize)

    # Create a pivot table for the heatmap
    times = sorted(pattern_df["id_time"].unique())
    units = sorted(pattern_df["unit_order"].unique())

    # Map status to numeric values: control=0, newly treated=1, already treated=2
    status_map = {"control": 0, "treated": 1, "already_treated": 2}
    pattern_df = pattern_df.copy()
    pattern_df["status_num"] = pattern_df["status"].map(status_map).fillna(0)

    # Create matrix
    matrix = np.full((len(units), len(times)), 0.0)
    time_idx = {t: i for i, t in enumerate(times)}
    unit_idx = {u: i for i, u in enumerate(units)}

    for _, row in pattern_df.iterrows():
        ti = time_idx.get(row["id_time"])
        ui = unit_idx.get(row["unit_order"])
        if ti is not None and ui is not None:
            matrix[ui, ti] = row["status_num"]

    # 3-color scheme: white (control), light blue (newly treated), dark blue (already treated)
    colors = ["#FFFFFF", "#6BAED6", "#08519C"]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    # Use pcolormesh for better grid control
    im = ax_.pcolormesh(
        np.arange(len(times) + 1) - 0.5,
        np.arange(len(units) + 1) - 0.5,
        matrix,
        cmap=cmap,
        norm=norm,
        edgecolors="#DDDDDD",
        linewidth=0.5,
    )

    # Axis ticks
    ax_.set_xticks(range(len(times)))
    ax_.set_xticklabels([str(int(t)) if t == int(t) else str(t) for t in times], fontsize=8)
    ax_.set_xlim(-0.5, len(times) - 0.5)
    ax_.set_ylim(-0.5, len(units) - 0.5)

    if len(units) <= 30:
        ax_.set_yticks(range(len(units)))
        ax_.set_yticklabels([str(u) for u in units], fontsize=7)
    else:
        # Sparse y ticks for many units
        step = max(1, len(units) // 10)
        ytick_positions = list(range(0, len(units), step))
        ax_.set_yticks(ytick_positions)
        ax_.set_yticklabels([str(units[i]) for i in ytick_positions], fontsize=7)

    # Legend via proxy artists - place outside/below the plot area
    legend_elements = [
        Patch(facecolor="#FFFFFF", edgecolor="#666666", linewidth=0.8, label="Control"),
        Patch(facecolor="#6BAED6", edgecolor="#666666", linewidth=0.8, label="Newly Treated"),
        Patch(facecolor="#08519C", edgecolor="#666666", linewidth=0.8, label="Already Treated"),
    ]
    ax_.legend(
        handles=legend_elements,
        frameon=True,
        fontsize=9,
        loc="lower right",
        edgecolor="#CCCCCC",
        fancybox=False,
    )

    # Labels
    default_title = "Treatment Timing Pattern"
    default_xlabel = "Time Period"
    default_ylabel = "Unit (ordered by treatment time)"

    if style == "publication":
        _apply_publication_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )
    else:
        _apply_default_style(
            ax_,
            title=title or default_title,
            xlabel=xlabel or default_xlabel,
            ylabel=ylabel or default_ylabel,
        )

    return _finalize_figure(fig, ax_, save=save, dpi=dpi, show=show, style=style, _skip_layout=_external)


def plot_diagnostics(
    check_result: DidCheckResult,
    *,
    result: DidResult | None = None,
    panels: str = "auto",
    ci: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    style: str = "publication",
    ci_level: float = 0.90,
    save: str | None = None,
    dpi: int = 150,
    show: bool = True,
) -> Any:
    """Plot combined diagnostic panel.

    By default displays two panels (trends + placebo) or three panels
    (trends + estimates + placebo) when a fitted *result* is provided.

    Parameters
    ----------
    check_result : DidCheckResult
        Diagnostic check result.
    result : DidResult, optional
        If provided, an estimates panel is included in the middle.
    panels : str
        Panel layout mode:

        - ``"auto"``: two panels unless *result* is given, in which case three.
        - ``"trends+placebo"``: always two panels (trends/pattern + placebo).
        - ``"trends+estimates+placebo"``: always three panels (requires *result*).
    ci : bool
        Whether to show confidence bands on the trends subplot.
    title : str, optional
        Overall figure title.
    figsize : tuple, optional
        Figure dimensions. Defaults to ``(12, 5)`` for two panels or
        ``(15, 5)`` for three panels.
    style : str
        ``"publication"`` or ``"default"``.
    ci_level : float
        Confidence level.
    save : str, optional
        File path to save figure.
    dpi : int
        Resolution.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, plt = _require_matplotlib()

    if style == "publication":
        _set_publication_font()

    # Determine panel count
    valid_panels = ("auto", "trends+placebo", "trends+estimates+placebo")
    if panels not in valid_panels:
        raise ValueError(f"panels must be one of {valid_panels}, got {panels!r}")

    if panels == "auto":
        use_three = result is not None
    elif panels == "trends+estimates+placebo":
        if result is None:
            raise ValueError(
                "panels='trends+estimates+placebo' requires a DidResult via the result= parameter."
            )
        use_three = True
    else:
        use_three = False

    # Detect SA design (pattern table present, no trends)
    is_sa = bool(check_result.pattern_table) and not check_result.trends_table

    # Figure layout
    n_panels = 3 if use_three else 2
    if figsize is None:
        figsize = (15, 5) if use_three else (12, 5)

    if use_three:
        width_ratios = [1.2, 1, 1]
    else:
        width_ratios = [1.2, 1]

    fig, axes = plt.subplots(
        1, n_panels, figsize=figsize,
        gridspec_kw={"width_ratios": width_ratios},
    )

    # -- Left panel: trends or pattern (SA) --
    if is_sa:
        try:
            plot_pattern(
                check_result,
                style=style,
                ax=axes[0],
                show=False,
            )
        except (ValueError, Exception):
            axes[0].text(
                0.5, 0.5, "No pattern data",
                ha="center", va="center", transform=axes[0].transAxes, fontsize=10,
            )
            axes[0].set_axis_off()
    else:
        try:
            plot_trends(
                check_result,
                ci=ci,
                style=style,
                ci_level=ci_level,
                ax=axes[0],
                show=False,
            )
        except ValueError:
            axes[0].text(
                0.5, 0.5, "No trend data",
                ha="center", va="center", transform=axes[0].transAxes, fontsize=10,
            )
            axes[0].set_axis_off()

    # -- Middle panel (if three-panel): estimates --
    if use_three:
        try:
            plot_estimates(
                result,
                check_fit=check_result,
                style=style,
                ci_level=ci_level,
                ax=axes[1],
                show=False,
            )
        except (ValueError, Exception):
            # Fallback: plot without check_fit overlay
            try:
                plot_estimates(
                    result,
                    style=style,
                    ci_level=ci_level,
                    ax=axes[1],
                    show=False,
                )
            except (ValueError, Exception):
                axes[1].text(
                    0.5, 0.5, "Estimates not available",
                    ha="center", va="center", transform=axes[1].transAxes, fontsize=10,
                )
                axes[1].set_axis_off()

    # -- Right panel: placebo --
    placebo_idx = 2 if use_three else 1
    try:
        plot_placebo(
            check_result,
            style=style,
            ci_level=ci_level,
            ax=axes[placebo_idx],
            show=False,
        )
    except ValueError:
        axes[placebo_idx].text(
            0.5, 0.5, "No placebo data",
            ha="center", va="center", transform=axes[placebo_idx].transAxes, fontsize=10,
        )
        axes[placebo_idx].set_axis_off()

    # Suptitle with proper spacing
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    fig.subplots_adjust(wspace=0.35)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return fig


__all__ = [
    "plot_estimates",
    "plot_trends",
    "plot_placebo",
    "plot_pattern",
    "plot_diagnostics",
]
