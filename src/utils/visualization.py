# src/utils/visualization.py
from __future__ import annotations


import math
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from typing import Dict, Iterable, Optional
import matplotlib.pyplot as plt





def plot_acf_scaled(
    series: pd.Series,
    ax,
    title: str,
    lags: int = 50,
    conf_level: float = 0.95,
    abs_series: bool = False,   # <-- NEW
):
    """
    Plot ACF with y-axis scaled tightly to the data.

    If abs_series=True, plots ACF of |series| (useful for volatility clustering).
    """
    series = series.dropna()

    if abs_series:
        series = series.abs()
        title = f"|{title}|"  # minimal cue in the subplot title

    plot_acf(
        series,
        lags=lags,
        ax=ax,
        zero=False,
        alpha=1 - conf_level,
    )

    # Tighten y-axis around actual values
    lines = ax.lines
    y_vals = np.concatenate(
        [line.get_ydata() for line in lines if len(line.get_ydata()) > 0]
    )

    if y_vals.size > 0 and np.all(np.isfinite(y_vals)):
        ymax = float(np.max(np.abs(y_vals)))
        if ymax > 0:
            ax.set_ylim(-1.2 * ymax, 1.2 * ymax)

    ax.set_title(title)

















def plot_acf_scaled_grid(
    data: pd.DataFrame,
    *,
    lags: int = 50,
    conf_level: float = 0.95,
    cols: int = 3,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    max_assets: Optional[int] = None,
    dropna: str = "any",
    abs_series: bool = False,   # <-- NEW
):
    """
    Plot scaled ACFs for many assets (columns) in a subplot grid,
    using plot_acf_scaled() for each series.

    If abs_series=True, plots ACF of |series| for each column.
    """
    if data is None or data.empty:
        raise ValueError("data is empty")

    df = data.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    if dropna == "any":
        df = df.dropna(how="any")
    elif dropna == "all":
        df = df.dropna(how="all")
    else:
        raise ValueError("dropna must be 'any' or 'all'.")

    if df.empty:
        raise ValueError("data is empty after dropna")

    if max_assets is not None:
        df = df.iloc[:, :max_assets]

    n = df.shape[1]
    if n == 0:
        raise ValueError("No columns to plot")

    rows = math.ceil(n / cols)

    if figsize is None:
        figsize = (4.5 * cols, 3.0 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(df.columns):
        ax = axes[i]
        series = df[col].dropna()
        plot_acf_scaled(
            series=series,
            ax=ax,
            title=str(col),
            lags=lags,
            conf_level=conf_level,
            abs_series=abs_series,   # <-- pass through
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if title is None:
        title = "ACF(|x|) — Multi-Asset Overview" if abs_series else "ACF(x) — Multi-Asset Overview"
    fig.suptitle(title, y=0.995)

    plt.tight_layout()
    plt.show()


















# src/utils/visualizations/price_plots.py

def plot_price_overview(
    raw_prices: dict,
    *,
    price_col: str = "Adj Close",
    normalize: bool = True,
    join: str = "outer",         # <-- key change
    fill_method: str = "ffill",  # <-- key change
    drop_all_nan: bool = True,
    figsize=(12, 5),
    title: str = "Normalized Price Overview",
):
    """
    Plot normalized prices for a dict[ticker] -> OHLC dataframe.

    Robust to different trading calendars (US vs SE vs funds) by using outer join and ffill.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # build a panel of price series
    series = {}
    for tkr, df in raw_prices.items():
        if df is None or getattr(df, "empty", False):
            continue

        # handle MultiIndex columns or weird column sets gracefully
        col = price_col if price_col in df.columns else ("Close" if "Close" in df.columns else None)
        if col is None:
            continue

        s = df[col].astype(float).copy()
        s.name = tkr
        series[tkr] = s

    if not series:
        raise ValueError("No usable price series found.")

    prices = pd.concat(series.values(), axis=1, join=join).sort_index()

    if fill_method == "ffill":
        prices = prices.ffill()
    elif fill_method == "bfill":
        prices = prices.bfill()
    elif fill_method is None:
        pass
    else:
        raise ValueError("fill_method must be 'ffill', 'bfill', or None")

    if drop_all_nan:
        prices = prices.dropna(how="all")

    if normalize:
        # normalize each column by its first non-NaN value (not a shared date)
        base = prices.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
        prices = prices.divide(base, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    prices.plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price" if normalize else "Price")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

    return prices
















# src/utils/visualization.py


def plot_log_returns_overview(
    log_returns: pd.DataFrame,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    figsize=(12, 8),
    title: Optional[str] = None,
    alpha: float = 0.9,
    linewidth: float = 1.0,
    sharex: bool = True,
    grid: bool = True,
):
    """
    Plot log returns in separate stacked subplots (one per ticker).

    Parameters
    ----------
    log_returns : pd.DataFrame
        DataFrame of log returns with columns=tickers and DateTimeIndex.
    start, end : str or None
        Optional date filters.
    figsize : tuple
        Figure size.
    title : str or None
        Overall figure title.
    alpha : float
        Line transparency.
    linewidth : float
        Line width.
    sharex : bool
        Share x-axis across subplots.
    grid : bool
        Show grid.
    """
    if log_returns is None or log_returns.empty:
        raise ValueError("log_returns is empty")

    df = log_returns.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError("log_returns is empty after date filtering")

    n_assets = df.shape[1]

    fig, axes = plt.subplots(
        nrows=n_assets,
        ncols=1,
        figsize=figsize,
        sharex=sharex,
    )

    # Handle single-asset case
    if n_assets == 1:
        axes = [axes]

    for ax, col in zip(axes, df.columns):
        ax.plot(
            df.index,
            df[col].values,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.set_title(col, fontsize=10)
        ax.axhline(0.0, linewidth=1.0, alpha=0.6)

        if grid:
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")

    if title is None:
        title = "Log Returns Overview"
    fig.suptitle(title, y=0.995)

    plt.tight_layout()
    plt.show()

    return df




























def plot_vol_and_resid_grid(
    vol_by_asset: Dict[str, pd.Series],
    z_by_asset: Optional[Dict[str, pd.Series]] = None,
    *,
    show_resid: bool = True,
    cols: int = 2,
    max_assets: Optional[int] = None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    sharex: bool = True,
    grid: bool = True,
    alpha: float = 0.9,
    linewidth: float = 1.2,
):
    """
    Plot a grid overview of conditional volatility (sigma_t) and optional standardized residuals (z_t).

    Parameters
    ----------
    vol_by_asset : dict[str, pd.Series]
        {asset -> sigma_t series}
    z_by_asset : dict[str, pd.Series] or None
        {asset -> z_t series}. Required if show_resid=True.
    show_resid : bool
        If True, show residual series below volatility for each asset.
        If False, only volatility is shown (cleaner).
    cols : int
        Number of assets per row (grid columns).
    max_assets : int or None
        If set, only plot first max_assets assets (prevents giant figures).
    figsize : tuple or None
        Auto-sized if None.
    title : str or None
        Overall title.
    sharex : bool
        Share x-axis across subplots.
    grid : bool
        Show subplot gridlines.
    alpha, linewidth : float
        Line style.
    """
    if not vol_by_asset:
        raise ValueError("vol_by_asset is empty")

    if show_resid and not z_by_asset:
        raise ValueError("show_resid=True requires z_by_asset")

    # Deterministic ordering
    assets = sorted(vol_by_asset.keys())
    if max_assets is not None:
        assets = assets[:max_assets]

    n_assets = len(assets)
    if n_assets == 0:
        raise ValueError("No assets to plot after max_assets filtering")

    # Each asset uses 1 row (vol only) or 2 rows (vol + resid)
    rows_per_asset = 2 if show_resid else 1

    # Grid layout: assets arranged in `cols` columns
    asset_rows = math.ceil(n_assets / cols)
    total_rows = asset_rows * rows_per_asset

    if figsize is None:
        # heuristic sizing
        fig_w = 5.5 * cols
        fig_h = 2.6 * total_rows
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=cols,
        figsize=figsize,
        sharex=sharex,
    )

    # Normalize axes to 2D array [row, col]
    if total_rows == 1 and cols == 1:
        axes = [[axes]]
    elif total_rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Helper to pick the top-left axis for an asset cell
    def _cell_axes(asset_idx: int):
        r_block = (asset_idx // cols) * rows_per_asset
        c = asset_idx % cols
        ax_vol = axes[r_block][c]
        ax_res = axes[r_block + 1][c] if show_resid else None
        return ax_vol, ax_res

    # Plot each asset
    for i, asset in enumerate(assets):
        ax_vol, ax_res = _cell_axes(i)

        sigma = pd.Series(vol_by_asset[asset]).dropna()
        if not isinstance(sigma.index, pd.DatetimeIndex):
            try:
                sigma.index = pd.to_datetime(sigma.index)
            except Exception:
                pass
        sigma = sigma.sort_index()

        ax_vol.plot(sigma.index, sigma.values, alpha=alpha, linewidth=linewidth)
        ax_vol.set_title(f"{asset} — σₜ", fontsize=10)
        if grid:
            ax_vol.grid(True, alpha=0.3)

        if show_resid:
            z = pd.Series(z_by_asset[asset]).dropna()
            if not isinstance(z.index, pd.DatetimeIndex):
                try:
                    z.index = pd.to_datetime(z.index)
                except Exception:
                    pass
            z = z.sort_index()

            ax_res.plot(z.index, z.values, alpha=alpha, linewidth=linewidth)
            ax_res.axhline(0.0, linewidth=1.0, alpha=0.6)
            ax_res.set_title(f"{asset} — zₜ", fontsize=10)
            if grid:
                ax_res.grid(True, alpha=0.3)

    # Turn off unused cells (both vol/res rows if show_resid)
    for j in range(n_assets, asset_rows * cols):
        ax_vol, ax_res = _cell_axes(j)
        ax_vol.axis("off")
        if ax_res is not None:
            ax_res.axis("off")

    if title is None:
        title = "Conditional Volatility (and Standardized Residuals)" if show_resid else "Conditional Volatility"
    fig.suptitle(title, y=0.995)

    plt.tight_layout()
    plt.show()



















def plot_corr_heatmap(
    corr: pd.DataFrame,
    *,
    title: str,
    figsize=(8, 6),
    annot: bool = False,
    fmt: str = ".2f",
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_colorbar: bool = True,
):
    """
    Plot a single correlation heatmap using matplotlib.

    Parameters
    ----------
    corr : pd.DataFrame
        Square correlation matrix.
    annot : bool
        If True, write numeric values in each cell (best for small matrices).
    """
    if corr is None or corr.empty:
        raise ValueError("corr is empty")

    labels = list(corr.columns)
    mat = corr.values

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")

    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()















def plot_corr_heatmaps(
    pearson: pd.DataFrame,
    spearman: pd.DataFrame,
    kendall: pd.DataFrame,
    *,
    figsize=(8, 6),
    annot: bool = False,
):
    """
    Plot Pearson, Spearman, and Kendall correlation heatmaps.
    """
    plot_corr_heatmap(pearson, title="Pearson correlation", figsize=figsize, annot=annot)
    plot_corr_heatmap(spearman, title="Spearman correlation", figsize=figsize, annot=annot)
    plot_corr_heatmap(kendall, title="Kendall correlation", figsize=figsize, annot=annot)






















def plot_return_vs_risk(
    points: pd.DataFrame,
    *,
    rf_annual: Optional[float] = None,
    title: str = "Return vs Risk",
    figsize=(8, 6),
    annotate: bool = True,
):
    """
    Scatter plot: annualized volatility (x) vs annualized return (y).
    Expects points with columns: ['ann_return', 'ann_vol'] and an optional 'PORTFOLIO' row.
    """
    if points is None or points.empty:
        raise ValueError("points is empty")
    if "ann_return" not in points.columns or "ann_vol" not in points.columns:
        raise ValueError("points must contain columns ['ann_return','ann_vol']")

    fig, ax = plt.subplots(figsize=figsize)

    assets = points.drop(index=["PORTFOLIO"], errors="ignore")
    ax.scatter(assets["ann_vol"], assets["ann_return"])

    if annotate:
        for name, row in assets.iterrows():
            ax.annotate(str(name), (row["ann_vol"], row["ann_return"]),
                        fontsize=9, xytext=(5, 3), textcoords="offset points")

    if "PORTFOLIO" in points.index:
        p = points.loc["PORTFOLIO"]
        ax.scatter([p["ann_vol"]], [p["ann_return"]], marker="x", s=120)
        ax.annotate("PORTFOLIO", (p["ann_vol"], p["ann_return"]),
                    fontsize=10, xytext=(7, 5), textcoords="offset points")

    if rf_annual is not None:
        ax.axhline(rf_annual, linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return (log-return approx.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()














def plot_return_vs_risk_progression(
    points_long: pd.DataFrame,
    *,
    rf_annual: Optional[float] = None,
    title: str = "Return vs Risk (Grid)",
    figsize=(12, 8),
    cols: int = 3,                 # NEW: how many subplots per row
    plot_assets: bool = True,
    plot_portfolio: bool = True,
    annotate: bool = False,        # NEW: annotate each subplot (off by default)
    alpha: float = 0.9,
    marker_size: float = 35.0,
    portfolio_marker_size: float = 120.0,
    pad_frac: float = 0.06,        # NEW: padding around global limits
):
    if points_long is None or points_long.empty:
        raise ValueError("points_long is empty")

    df = points_long.copy()

    window_ids = np.array(sorted(df["window_idx"].unique()), dtype=int)
    if window_ids.size == 0:
        raise ValueError("No window_idx found")

    # Base colors per asset (constant across all windows)
    assets_all = sorted(df["asset"].unique())
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    base_color_map = {a: prop_cycle[i % len(prop_cycle)] for i, a in enumerate(assets_all)}

    # --- Compute GLOBAL axis limits so every subplot matches ---
    dff = df.copy()
    if not plot_portfolio:
        dff = dff[dff["asset"] != "PORTFOLIO"]
    if not plot_assets:
        dff = dff[dff["asset"] == "PORTFOLIO"]

    if dff.empty:
        raise ValueError("Nothing to plot after plot_assets/plot_portfolio filtering")

    x_min, x_max = float(dff["ann_vol"].min()), float(dff["ann_vol"].max())
    y_min, y_max = float(dff["ann_return"].min()), float(dff["ann_return"].max())

    # add padding
    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = pad_frac * (x_span if x_span > 0 else max(abs(x_max), 1e-8))
    y_pad = pad_frac * (y_span if y_span > 0 else max(abs(y_max), 1e-8))

    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # --- Make grid ---
    n = int(window_ids.size)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes).reshape(rows, cols)

    for idx, w in enumerate(window_ids):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        win = df[df["window_idx"] == w]

        if rf_annual is not None:
            ax.axhline(rf_annual, linewidth=1.0, alpha=0.8)

        # Assets
        if plot_assets:
            assets = win[win["asset"] != "PORTFOLIO"]
            for asset, sub in assets.groupby("asset"):
                ax.scatter(
                    sub["ann_vol"].values,
                    sub["ann_return"].values,
                    s=marker_size,
                    alpha=alpha,
                    color=base_color_map.get(asset, "C0"),
                )

        # Portfolio
        if plot_portfolio:
            port = win[win["asset"] == "PORTFOLIO"]
            if not port.empty:
                ax.scatter(
                    port["ann_vol"].values,
                    port["ann_return"].values,
                    marker="x",
                    s=portfolio_marker_size,
                    alpha=alpha,
                    color=base_color_map.get("PORTFOLIO", "C3"),
                )

        # constant axes
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.grid(True, alpha=0.3)
        ax.set_title(f"Window {w}", fontsize=10)

        if annotate:
            for _, row in win.iterrows():
                ax.annotate(
                    str(row["asset"]),
                    (row["ann_vol"], row["ann_return"]),
                    fontsize=8,
                    xytext=(4, 2),
                    textcoords="offset points",
                )

    # turn off unused subplots
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")

    # shared labels (only on outer edges)
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Annualized Volatility")
    for ax in axes[:, 0]:
        if ax.has_data():
            ax.set_ylabel("Annualized Return (log-return approx.)")

    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.show()




















# src/utils/visualization.py




# def plot_portfolio_comparison_return_risk(
#     portfolio_points: pd.DataFrame,
#     *,
#     rf_annual: Optional[float] = None,
#     title: str = "Portfolio comparison: Return vs Risk",
#     figsize=(8, 6),
#     annotate: bool = True,
# ):
#     """
#     portfolio_points: index=portfolio name, columns include ['ann_return','ann_vol'].
#     """
#     if portfolio_points is None or portfolio_points.empty:
#         raise ValueError("portfolio_points is empty")

#     fig, ax = plt.subplots(figsize=figsize)

#     ax.scatter(portfolio_points["ann_vol"], portfolio_points["ann_return"], marker="x", s=140)

#     if annotate:
#         for name, row in portfolio_points.iterrows():
#             ax.annotate(str(name), (row["ann_vol"], row["ann_return"]),
#                         fontsize=10, xytext=(7, 5), textcoords="offset points")

#     if rf_annual is not None:
#         ax.axhline(rf_annual, linewidth=1.0, alpha=0.8)

#     ax.set_xlabel("Annualized Volatility")
#     ax.set_ylabel("Annualized Return (log-return approx.)")
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()




def plot_portfolio_comparison_return_risk(
    portfolio_points: pd.DataFrame,
    *,
    rf_annual: Optional[float] = None,
    title: str = "Portfolio comparison: Return vs Risk",
    figsize=(8, 6),
    annotate: bool = True,
    # --- NEW: optionally compute + show correlation heatmaps beside the scatter ---
    z_panel: Optional[pd.DataFrame] = None,
    corr_assets: Optional[Iterable[str]] = None,
    show_corr: bool = True,
    corr_annot: bool = False,
    corr_fmt: str = ".2f",
    corr_vmin: float = -1.0,
    corr_vmax: float = 1.0,
):
    """
    Scatter plot: annualized volatility (x) vs annualized return (y) for portfolio alternatives.

    If z_panel is provided (standardized residuals panel), also shows Pearson/Spearman/Kendall
    correlation heatmaps in the same figure (stacked on the right).

    Parameters
    ----------
    portfolio_points : pd.DataFrame
        index=portfolio name, columns include ['ann_return','ann_vol'].
    z_panel : pd.DataFrame, optional
        Standardized residuals panel (date × asset). Correlations computed from this.
    corr_assets : Iterable[str], optional
        Subset of assets/columns to include in correlation heatmaps. Defaults to all z_panel columns.
    """
    if portfolio_points is None or portfolio_points.empty:
        raise ValueError("portfolio_points is empty")
    if "ann_vol" not in portfolio_points.columns or "ann_return" not in portfolio_points.columns:
        raise ValueError("portfolio_points must have columns ['ann_vol','ann_return']")

    # --- If no correlations requested / available, keep original behavior ---
    use_corr = bool(show_corr and z_panel is not None and isinstance(z_panel, pd.DataFrame) and not z_panel.empty)

    if not use_corr:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(portfolio_points["ann_vol"], portfolio_points["ann_return"], s=70)

        if annotate:
            for name, row in portfolio_points.iterrows():
                ax.annotate(
                    str(name),
                    (row["ann_vol"], row["ann_return"]),
                    fontsize=10,
                    xytext=(7, 5),
                    textcoords="offset points",
                )

        if rf_annual is not None:
            ax.axhline(rf_annual, linewidth=1.0, alpha=0.8)

        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return (log-return approx.)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # --- Compute correlations from z_panel ---
    if corr_assets is not None:
        corr_assets = [a for a in corr_assets if a in z_panel.columns]
        z_use = z_panel[corr_assets].dropna(how="all")
    else:
        z_use = z_panel.dropna(how="all")

    pearson = z_use.corr(method="pearson")
    spearman = z_use.corr(method="spearman")
    kendall = z_use.corr(method="kendall")

    import matplotlib.gridspec as gridspec

    # Better default size when adding 3 heatmaps
    if figsize == (8, 6):
        figsize = (14, 8)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1, 1, 1],
        wspace=0.25,
        hspace=0.35,
    )

    # --- Left: scatter spanning all rows ---
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_scatter.scatter(portfolio_points["ann_vol"], portfolio_points["ann_return"], s=70)

    if annotate:
        for name, row in portfolio_points.iterrows():
            ax_scatter.annotate(
                str(name),
                (row["ann_vol"], row["ann_return"]),
                fontsize=10,
                xytext=(7, 5),
                textcoords="offset points",
            )

    if rf_annual is not None:
        ax_scatter.axhline(rf_annual, linewidth=1.0, alpha=0.8)

    ax_scatter.set_xlabel("Annualized Volatility")
    ax_scatter.set_ylabel("Annualized Return (log-return approx.)")
    ax_scatter.set_title(title)
    ax_scatter.grid(True, alpha=0.3)

    # --- Right: heatmaps stacked (Pearson/Spearman/Kendall) ---
    def _heatmap(ax, corr: pd.DataFrame, htitle: str):
        labels = list(corr.columns)
        mat = corr.values

        im = ax.imshow(mat, vmin=corr_vmin, vmax=corr_vmax, cmap="RdYlGn", aspect="auto")
        ax.set_title(htitle)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        if corr_annot:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, format(val, corr_fmt), ha="center", va="center", fontsize=8)

        # individual colorbar like your existing plot_corr_heatmap
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation")
        return im

    ax_p = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 1])
    ax_k = fig.add_subplot(gs[2, 1])

    _heatmap(ax_p, pearson, "Pearson correlation")
    _heatmap(ax_s, spearman, "Spearman correlation")
    _heatmap(ax_k, kendall, "Kendall correlation")

    plt.tight_layout()
    plt.show()









def plot_section5_portfolio_evaluation_rows(
    port_points: pd.DataFrame,
    log_returns: pd.DataFrame,
    z_panel: pd.DataFrame,
    portfolios: dict,
    *,
    rf_annual: float,
    start: str,
    end: str,
    current_name: str = "Current",
    show_performance: bool = True,
    corr_methods: tuple[str, str, str] = ("pearson", "spearman", "kendall"),
    corr_vmin: float = -1.0,
    corr_vmax: float = 1.0,
):
    """
    Row-per-alternative evaluation.

    For each alternative portfolio P (excluding current_name), produce a row that compares:
      - Markowitz point: Current vs P (only 2 points)
      - Correlations on assets in (Current assets ∪ P assets), for pearson/spearman/kendall
      - Optional: Normalized performance (Current vs P)
    """

    if current_name not in portfolios:
        raise ValueError(f"current_name='{current_name}' not found in portfolios")

    current_assets = set(portfolios[current_name].keys())
    alt_names = [k for k in portfolios.keys() if k != current_name]
    if not alt_names:
        raise ValueError("No alternative portfolios found (portfolios only contains Current)")

    # slice returns to window once
    r = log_returns.copy()
    if start is not None:
        r = r.loc[r.index >= start]
    if end is not None:
        r = r.loc[r.index <= end]

    # determine figure layout
    n_rows = len(alt_names)
    # columns: 1 markowitz + 3 corr + optional performance
    n_corr = len(corr_methods)
    n_cols = 1 + n_corr + (1 if show_performance else 0)

    fig_w = 4.8 * n_cols
    fig_h = 3.6 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    if n_rows == 1:
        axes = np.array([axes])  # ensure 2D indexing

    # helper: plot 2-point markowitz for a pair
    def _plot_pair_markowitz(ax, name_alt: str):
        # expects port_points indexed by portfolio name
        if name_alt not in port_points.index or current_name not in port_points.index:
            ax.axis("off")
            ax.set_title("Missing port_points")
            return

        p_cur = port_points.loc[current_name]
        p_alt = port_points.loc[name_alt]

        ax.scatter([p_cur["ann_vol"]], [p_cur["ann_return"]], s=80, marker="o", label=current_name)
        ax.scatter([p_alt["ann_vol"]], [p_alt["ann_return"]], s=80, marker="x", label=name_alt)

        if rf_annual is not None:
            ax.axhline(rf_annual, linewidth=1.0, alpha=0.7)

        ax.set_xlabel("Vol")
        ax.set_ylabel("Ret")
        ax.set_title("Markowitz (2 points)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # helper: plot corr heatmap
    def _plot_corr(ax, corr_df: pd.DataFrame, title: str):
        labels = list(corr_df.columns)
        mat = corr_df.values
        im = ax.imshow(mat, vmin=corr_vmin, vmax=corr_vmax, cmap="RdYlGn", aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # helper: performance plot
    def _plot_pair_performance(ax, name_alt: str):
        w_cur = pd.Series(portfolios[current_name], dtype=float)
        w_alt = pd.Series(portfolios[name_alt], dtype=float)

        # require both portfolios' assets exist in returns
        cur_assets = list(w_cur.index)
        alt_assets = list(w_alt.index)

        miss_cur = [a for a in cur_assets if a not in r.columns]
        miss_alt = [a for a in alt_assets if a not in r.columns]
        if miss_cur or miss_alt:
            ax.axis("off")
            ax.set_title("Missing returns cols")
            return

        cur = np.exp((r[cur_assets] @ w_cur).cumsum())
        alt = np.exp((r[alt_assets] @ w_alt).cumsum())

        cur = cur / cur.iloc[0]
        alt = alt / alt.iloc[0]

        ax.plot(cur.index, cur.values, label=current_name, linewidth=2.0)
        ax.plot(alt.index, alt.values, label=name_alt, linewidth=1.5)

        ax.set_title("Normalized perf", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for i, alt_name in enumerate(alt_names):
        row_axes = axes[i, :]

        # Row label on left margin-ish
        row_axes[0].text(
            -0.25, 0.5, f"{alt_name} vs {current_name}",
            transform=row_axes[0].transAxes,
            rotation=90, va="center", ha="center", fontsize=11, fontweight="bold"
        )

        # --- 1) Markowitz 2-point plot
        _plot_pair_markowitz(row_axes[0], alt_name)

        # --- 2) Correlations for ONLY (Current ∪ Alt assets)
        assets = sorted(current_assets | set(portfolios[alt_name].keys()))
        assets = [a for a in assets if a in z_panel.columns]
        if len(assets) < 2:
            # if not enough assets, blank corr panels
            for j in range(n_corr):
                ax = row_axes[1 + j]
                ax.axis("off")
                ax.set_title("Corr unavailable")
        else:
            z_sub = z_panel[assets].dropna(how="all")
            for j, method in enumerate(corr_methods):
                corr = z_sub.corr(method=method)
                _plot_corr(row_axes[1 + j], corr, f"{method.title()} corr")

        # --- 3) Optional performance (Current vs Alt)
        if show_performance:
            _plot_pair_performance(row_axes[-1], alt_name)

    plt.tight_layout()
    plt.show()
