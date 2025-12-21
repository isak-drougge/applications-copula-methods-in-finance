# src/utils/portfolio_analysis.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence, List
import numpy as np
import pandas as pd


@dataclass
class MarkowitzResult:
    points: pd.DataFrame          # rows: assets + PORTFOLIO, cols: ann_return, ann_vol
    mu_daily: pd.Series           # daily mean vector (log returns)
    Sigma_daily: pd.DataFrame     # daily covariance matrix
    w_risky: pd.Series            # risky weights aligned to columns
    w_rf: float                   # risk-free weight (1 - sum risky)
    rf_annual: float
    rf_daily: float


def _filter_returns(
    log_returns: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    *,
    dropna: str = "any",  # "any" for strict covariance
) -> pd.DataFrame:
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

    df = df.dropna(how=dropna)
    if df.empty:
        raise ValueError("No data left after filtering/dropna.")

    return df


def _get_risky_weights_and_rf_weight(
    columns: list[str],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, float]:
    """
    Risky weights can sum to <= 1. Remaining weight goes to risk-free.
    """
    w_risky = np.array([float(weights.get(c, 0.0)) for c in columns], dtype=float)
    s = float(w_risky.sum())

    if not np.isfinite(s) or s < 0:
        raise ValueError("Invalid weights.")
    if s > 1.0 + 1e-10:
        raise ValueError(f"Risky weights sum to {s:.6f} > 1. Set them to sum to <= 1.")

    w_rf = 1.0 - s
    return w_risky, w_rf


def compute_markowitz_points_with_rf(
    log_returns: pd.DataFrame,
    weights: Dict[str, float],
    *,
    rf_annual: float,
    start: Optional[str] = None,
    end: Optional[str] = None,
    trading_days: int = 252,
) -> MarkowitzResult:
    """
    Compute annualized return/vol points for each asset and for your portfolio.

    Uses log returns:
      E[annual log return] ≈ 252 * mean(daily log return)
      annual vol ≈ sqrt(252) * std(daily log return)

    Risk-free handling:
      w_rf = 1 - sum(risky weights)
      Portfolio mean includes w_rf * rf_daily
      Portfolio variance excludes risk-free (zero var/cov)
    """
    df = _filter_returns(log_returns, start, end, dropna="any")

    cols = list(df.columns)
    w_risky, w_rf = _get_risky_weights_and_rf_weight(cols, weights)

    mu_daily = df.mean()          # pd.Series
    Sigma_daily = df.cov()        # pd.DataFrame

    rf_daily = float(rf_annual) / float(trading_days)

    # Portfolio daily moments
    port_mu_daily = float(w_rf * rf_daily + np.dot(w_risky, mu_daily.values))
    port_var_daily = float(w_risky @ Sigma_daily.values @ w_risky)
    port_vol_daily = float(np.sqrt(max(port_var_daily, 0.0)))

    # Asset points
    points = pd.DataFrame(
        {
            "ann_return": mu_daily.values * trading_days,
            "ann_vol": np.sqrt(np.diag(Sigma_daily.values)) * np.sqrt(trading_days),
        },
        index=cols,
    )

    # Portfolio point
    points.loc["PORTFOLIO", "ann_return"] = port_mu_daily * trading_days
    points.loc["PORTFOLIO", "ann_vol"] = port_vol_daily * np.sqrt(trading_days)

    return MarkowitzResult(
        points=points,
        mu_daily=mu_daily,
        Sigma_daily=Sigma_daily,
        w_risky=pd.Series(w_risky, index=cols),
        w_rf=w_rf,
        rf_annual=float(rf_annual),
        rf_daily=rf_daily,
    )


def compute_sharpe(
    ann_return: float,
    ann_vol: float,
    rf_annual: float,
) -> float:
    """
    Sharpe ratio using annualized return/vol (log-return approximation).
    """
    if ann_vol <= 0 or not np.isfinite(ann_vol):
        return np.nan
    return float((ann_return - rf_annual) / ann_vol)












@dataclass
class WindowMarkowitzOutput:
    window_idx: int
    start: pd.Timestamp
    end: pd.Timestamp
    points: pd.DataFrame          # rows assets + PORTFOLIO, cols ann_return, ann_vol
    corr_pearson: pd.DataFrame
    corr_spearman: pd.DataFrame
    corr_kendall: pd.DataFrame


def _rf_split_weights(columns: List[str], weights: Dict[str, float]) -> Tuple[np.ndarray, float]:
    w = np.array([float(weights.get(c, 0.0)) for c in columns], dtype=float)
    s = float(w.sum())
    if not np.isfinite(s) or s < 0:
        raise ValueError("Invalid weights.")
    if s > 1.0 + 1e-10:
        raise ValueError(f"Risky weights sum to {s:.6f} > 1. Set them to sum to <= 1.")
    return w, 1.0 - s


def _cov_from_sigma_and_corr(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    # Sigma = D R D
    D = np.diag(sigma)
    return D @ corr @ D


def compute_windowed_markowitz_garch(
    *,
    returns_panel: pd.DataFrame,  # log returns
    sigma_panel: pd.DataFrame,    # conditional vol σ_t per asset (daily)
    z_panel: pd.DataFrame,        # standardized residuals z_t per asset
    weights: Dict[str, float],
    rf_annual: float,
    window_days: int = 252,
    step_days: int = 21,
    trading_days: int = 252,
    sigma_agg: str = "mean",      # "mean" or "last"
    corr_for_cov: str = "pearson" # which corr to use to build Σ_k for risk
) -> List[WindowMarkowitzOutput]:
    """
    For each rolling window:
      - mu_k from returns in window
      - correlations from z in window (Pearson/Spearman/Kendall)
      - sigma_k from sigma_panel in window (mean or last)
      - covariance Σ_k = D_k R_k D_k
      - compute asset points and portfolio point (with risk-free weight)

    Returns a list of WindowMarkowitzOutput.
    """
    if returns_panel is None or returns_panel.empty:
        raise ValueError("returns_panel is empty")
    if sigma_panel is None or sigma_panel.empty:
        raise ValueError("sigma_panel is empty")
    if z_panel is None or z_panel.empty:
        raise ValueError("z_panel is empty")

    # Align on common columns and dates
    cols = [c for c in returns_panel.columns if c in sigma_panel.columns and c in z_panel.columns]
    if not cols:
        raise ValueError("No overlapping assets between returns_panel, sigma_panel, and z_panel")

    r = returns_panel[cols].copy()
    s = sigma_panel[cols].copy()
    z = z_panel[cols].copy()

    # strict alignment by date for window slicing
    idx = r.index.intersection(s.index).intersection(z.index)
    r = r.loc[idx].dropna(how="any")
    s = s.loc[r.index].dropna(how="any")
    z = z.loc[r.index].dropna(how="any")

    if r.empty:
        raise ValueError("No aligned data after dropping NaNs.")

    w_risky, w_rf = _rf_split_weights(cols, weights)
    rf_daily = float(rf_annual) / float(trading_days)

    outputs: List[WindowMarkowitzOutput] = []

    T = len(r)
    start_i = 0
    window_idx = 0

    while start_i + window_days <= T:
        end_i = start_i + window_days
        r_win = r.iloc[start_i:end_i]
        s_win = s.iloc[start_i:end_i]
        z_win = z.iloc[start_i:end_i]

        t0 = r_win.index[0]
        t1 = r_win.index[-1]

        # mean return in window
        mu_daily = r_win.mean()

        # correlations on standardized residuals
        corr_p = z_win.corr(method="pearson")
        corr_s = z_win.corr(method="spearman")
        corr_k = z_win.corr(method="kendall")

        # choose corr for covariance
        if corr_for_cov == "pearson":
            R = corr_p.values
        elif corr_for_cov == "spearman":
            R = corr_s.values
        elif corr_for_cov == "kendall":
            R = corr_k.values
        else:
            raise ValueError("corr_for_cov must be 'pearson', 'spearman', or 'kendall'")

        # sigma aggregation
        if sigma_agg == "mean":
            sigma_daily = s_win.mean().values
        elif sigma_agg == "last":
            sigma_daily = s_win.iloc[-1].values
        else:
            raise ValueError("sigma_agg must be 'mean' or 'last'")

        sigma_daily = np.asarray(sigma_daily, dtype=float)

        # asset points (annualized)
        points = pd.DataFrame(index=cols, columns=["ann_return", "ann_vol"], dtype=float)
        points["ann_return"] = mu_daily.values * trading_days
        points["ann_vol"] = sigma_daily * np.sqrt(trading_days)

        # portfolio moments
        Sigma_daily = _cov_from_sigma_and_corr(sigma_daily, R)
        port_mu_daily = float(w_rf * rf_daily + w_risky @ mu_daily.values)
        port_var_daily = float(w_risky @ Sigma_daily @ w_risky)
        port_vol_daily = float(np.sqrt(max(port_var_daily, 0.0)))

        points.loc["PORTFOLIO", "ann_return"] = port_mu_daily * trading_days
        points.loc["PORTFOLIO", "ann_vol"] = port_vol_daily * np.sqrt(trading_days)

        outputs.append(
            WindowMarkowitzOutput(
                window_idx=window_idx,
                start=pd.Timestamp(t0),
                end=pd.Timestamp(t1),
                points=points,
                corr_pearson=corr_p,
                corr_spearman=corr_s,
                corr_kendall=corr_k,
            )
        )

        window_idx += 1
        start_i += step_days

    return outputs


def flatten_window_points(
    window_outputs: List[WindowMarkowitzOutput],
) -> pd.DataFrame:
    """
    Convert list of WindowMarkowitzOutput into a long DataFrame:
    columns: window_idx, start, end, asset, ann_return, ann_vol
    """
    rows = []
    for out in window_outputs:
        for asset, row in out.points.iterrows():
            rows.append(
                {
                    "window_idx": out.window_idx,
                    "start": out.start,
                    "end": out.end,
                    "asset": str(asset),
                    "ann_return": float(row["ann_return"]),
                    "ann_vol": float(row["ann_vol"]),
                }
            )
    return pd.DataFrame(rows)
