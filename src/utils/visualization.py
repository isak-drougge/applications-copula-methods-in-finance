# src/utils/visualization.py


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
    raw_prices: Dict[str, pd.DataFrame],
    *,
    price_col_candidates: Iterable[str] = (
        "Adj Close",
        "AdjClose",
        "Close",
        "close",
        "adjclose",
    ),
    normalize: bool = True,
    join: str = "outer",          # "outer" or "inner"
    base: float = 1.0,
    figsize=(12, 6),
    title: Optional[str] = None,
    alpha: float = 0.9,
    linewidth: float = 1.5,
    legend: bool = True,
):
    """
    Build and plot a normalized price panel from load_prices_many output.

    Parameters
    ----------
    raw_prices : dict[str, pd.DataFrame]
        Output of load_prices_many.
    price_col_candidates : iterable[str]
        Price column names to try (first match used).
    normalize : bool
        Normalize each series to `base` at its first valid observation.
    join : {"outer","inner"}
        How to align dates across assets.
    base : float
        Normalization base level (default 1.0).
    """

    if not raw_prices:
        raise ValueError("raw_prices is empty")

    series_list = []

    # --- build price panel ---
    for ticker, df in raw_prices.items():
        if df is None or df.empty:
            continue

        col = next((c for c in price_col_candidates if c in df.columns), None)
        if col is None:
            raise ValueError(
                f"{ticker}: none of {tuple(price_col_candidates)} found in columns "
                f"{list(df.columns)}"
            )

        s = df[col].copy()

        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)

        s = s.sort_index().dropna()

        if s.empty:
            continue

        if normalize:
            s = base * s / s.iloc[0]

        s.name = ticker
        series_list.append(s)

    if not series_list:
        raise RuntimeError("No valid price series after processing")

    price_panel = pd.concat(series_list, axis=1, join=join)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=figsize)

    for col in price_panel.columns:
        ax.plot(
            price_panel.index,
            price_panel[col],
            label=col,
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price" if normalize else "Price")

    if title is None:
        title = "Normalized Price Overview" if normalize else "Price Overview"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)

    if legend:
        ax.legend(ncol=2, fontsize=9)

    plt.tight_layout()
    plt.show()

    return price_panel















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











def _blend_with_white(rgb, t: float):
    """
    t in [0,1]:
      t=0 -> white
      t=1 -> original rgb
    """
    r, g, b = rgb
    return (1 - t) + t * r, (1 - t) + t * g, (1 - t) + t * b








def plot_return_vs_risk_progression(
    points_long: pd.DataFrame,
    *,
    rf_annual: Optional[float] = None,
    title: str = "Return vs Risk (Progression)",
    figsize=(8, 6),
    plot_assets: bool = True,
    plot_portfolio: bool = True,
    annotate_last: bool = True,
    alpha: float = 0.9,
    marker_size: float = 35.0,
    portfolio_marker_size: float = 120.0,
):
    if points_long is None or points_long.empty:
        raise ValueError("points_long is empty")

    df = points_long.copy()

    window_ids = np.array(sorted(df["window_idx"].unique()), dtype=int)
    if window_ids.size == 0:
        raise ValueError("No window_idx found")

    # progression strength in [0.25..0.95]
    if window_ids.size == 1:
        strength_map = {int(window_ids[0]): 0.85}
    else:
        strengths = np.linspace(0.25, 0.95, num=window_ids.size)
        strength_map = {int(w): float(s) for w, s in zip(window_ids, strengths)}

    # base colors per asset (matplotlib default cycle)
    assets_all = sorted(df["asset"].unique())
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_cycle:
        prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    base_color_map = {}
    for i, a in enumerate(assets_all):
        base_color_map[a] = prop_cycle[i % len(prop_cycle)]

    def _blend_with_white(rgb, t: float):
        r, g, b = rgb
        return (1 - t) + t * r, (1 - t) + t * g, (1 - t) + t * b

    fig, ax = plt.subplots(figsize=figsize)

    if rf_annual is not None:
        ax.axhline(rf_annual, linewidth=1.0, alpha=0.8)

    # plot each asset separately so it keeps its color across windows
    for asset in assets_all:
        sub = df[df["asset"] == asset]
        if asset == "PORTFOLIO" and not plot_portfolio:
            continue
        if asset != "PORTFOLIO" and not plot_assets:
            continue

        base = plt.matplotlib.colors.to_rgb(base_color_map[asset])

        for w in window_ids:
            strength = strength_map[int(w)]
            color = _blend_with_white(base, strength)

            win = sub[sub["window_idx"] == w]
            if win.empty:
                continue

            if asset == "PORTFOLIO":
                ax.scatter(
                    win["ann_vol"].values,
                    win["ann_return"].values,
                    marker="x",
                    s=portfolio_marker_size,
                    alpha=alpha,
                    color=[color],
                )
            else:
                ax.scatter(
                    win["ann_vol"].values,
                    win["ann_return"].values,
                    s=marker_size,
                    alpha=alpha,
                    color=[color],
                )

    # annotate last window only
    if annotate_last:
        last_w = int(window_ids.max())
        last = df[df["window_idx"] == last_w]

        if plot_assets:
            for _, row in last[last["asset"] != "PORTFOLIO"].iterrows():
                ax.annotate(
                    str(row["asset"]),
                    (row["ann_vol"], row["ann_return"]),
                    fontsize=9,
                    xytext=(5, 3),
                    textcoords="offset points",
                )

        if plot_portfolio:
            port_last = last[last["asset"] == "PORTFOLIO"]
            if not port_last.empty:
                r0 = port_last.iloc[0]
                ax.annotate(
                    "PORTFOLIO",
                    (r0["ann_vol"], r0["ann_return"]),
                    fontsize=10,
                    xytext=(7, 5),
                    textcoords="offset points",
                )

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return (log-return approx.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
