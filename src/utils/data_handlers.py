from __future__ import annotations

from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from arch import arch_model












def compute_log_returns(
    raw_prices: Dict[str, pd.DataFrame],
    *,
    price_col_candidates: Iterable[str] = (
        "Adj Close",
        "AdjClose",
        "Close",
        "close",
        "adjclose",
    ),
    join: str = "outer",            # "outer" (union of dates) or "inner" (intersection)
    dropna: str = "any",            # "any" or "all" after return computation
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute log returns from load_prices_many output.

    Parameters
    ----------
    raw_prices : dict[str, pd.DataFrame]
        Output of load_prices_many: {ticker -> price dataframe}.
    price_col_candidates : iterable[str]
        Column names to try for the price series.
    join : {"outer","inner"}
        How to align the price panel across tickers before computing returns.
    dropna : {"any","all"}
        - "any": drop rows where any ticker return is NaN (strict; good for multivariate models)
        - "all": drop rows where all ticker returns are NaN (lenient; good for plotting)
    start, end : str or None
        Optional date filter applied to the price panel before returns are computed.

    Returns
    -------
    log_rets : pd.DataFrame
        DataFrame of log returns with columns=tickers and DateTimeIndex.
    """
    if not raw_prices:
        raise ValueError("raw_prices is empty")

    # Build aligned price panel
    series_list = []

    for ticker, df in raw_prices.items():
        if df is None or df.empty:
            continue

        col = next((c for c in price_col_candidates if c in df.columns), None)
        if col is None:
            raise ValueError(
                f"{ticker}: none of {tuple(price_col_candidates)} found in columns {list(df.columns)}"
            )

        s = df[col].copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s = s.sort_index()

        # optional filter window
        if start is not None:
            s = s[s.index >= pd.to_datetime(start)]
        if end is not None:
            s = s[s.index <= pd.to_datetime(end)]

        s = s.dropna()
        if s.empty:
            continue

        s.name = ticker
        series_list.append(s)

    if not series_list:
        raise RuntimeError("No valid price series found to compute returns.")

    prices = pd.concat(series_list, axis=1, join=join).sort_index()

    # Compute log returns
    log_prices = np.log(prices)
    log_rets = log_prices.diff()

    # Drop rows according to preference
    if dropna == "any":
        log_rets = log_rets.dropna(how="any")
    elif dropna == "all":
        log_rets = log_rets.dropna(how="all")
    else:
        raise ValueError("dropna must be 'any' or 'all'.")

    return log_rets















def mean_test_t(
    x: pd.Series,
) -> pd.DataFrame:
    """
    One-sample t-test summary for mean != 0 (like your notebook cells),
    returned as a 1-row DataFrame.
    """
    arr = pd.Series(x).astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        raise ValueError("Not enough observations for mean test.")

    mu_hat = float(arr.mean())
    sigma_hat = float(arr.std(ddof=1))
    se = sigma_hat / np.sqrt(n)
    t_stat = mu_hat / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else np.nan

    return pd.DataFrame(
        {
            "Estimate": [mu_hat],
            "Std. Error": [se],
            "t-stat": [t_stat],
            "p-value": [p_value],
            "N": [n],
        },
        index=["Mean return"],
    )

















def mean_tests_for_assets(
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run mean_test_t for each column in a returns DataFrame.

    Returns a DataFrame indexed by asset/ticker with columns:
    Estimate, Std. Error, t-stat, p-value, N.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df is empty")

    rows = []
    for col in returns_df.columns:
        res = mean_test_t(returns_df[col])
        row = res.iloc[0].to_dict()
        row["Asset"] = str(col)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("Asset")
    return out.sort_index()
















@dataclass(frozen=True)
class VolModelSpec:
    vol: str  # "ARCH" or "GARCH"
    p: int
    q: int

    def name(self) -> str:
        if self.vol.upper() == "ARCH":
            return f"ARCH({self.p})"
        return f"GARCH({self.p},{self.q})"




DEFAULT_MODEL_GRID: Tuple[VolModelSpec, ...] = (
    VolModelSpec("ARCH", 1, 0),
    VolModelSpec("ARCH", 2, 0),
    VolModelSpec("ARCH", 3, 0),
    VolModelSpec("GARCH", 1, 1),
    VolModelSpec("GARCH", 1, 2),
    VolModelSpec("GARCH", 2, 1),
)

DEFAULT_DISTS: Tuple[str, ...] = ("normal", "t")








def fit_vol_models_one_asset(
    returns: pd.Series,
    *,
    asset_name: str,
    model_grid: Iterable[VolModelSpec] = DEFAULT_MODEL_GRID,
    dists: Iterable[str] = DEFAULT_DISTS,
    mean: str = "Constant",              # "Constant", "Zero", "AR", etc.
    ar_lags: int = 0,                    # if mean="AR", set ar_lags>0
    ljungbox_lags: int = 10,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], object]]:
    """
    Grid search ARCH/GARCH specs and return:
      - results table (sorted by AIC)
      - dict of fitted model results keyed by (model_name, dist)

    Uses Ljung-Box on standardized residuals z and z^2 at one lag length.
    """
    x = pd.Series(returns).astype(float).dropna()
    x = x[np.isfinite(x)]
    if x.size < 50:
        raise ValueError(f"{asset_name}: too few observations ({x.size}) to fit ARCH/GARCH reliably")

    rows = []
    fitted_models: Dict[Tuple[str, str], object] = {}

    for spec in model_grid:
        for dist in dists:
            try:
                am = arch_model(
                    x,
                    mean=mean,
                    lags=ar_lags if mean.lower() == "ar" else None,
                    vol=spec.vol,
                    p=spec.p,
                    q=spec.q,
                    dist=dist,
                )
                res = am.fit(disp="off")

                z = pd.Series(res.std_resid).dropna()

                lb_z = acorr_ljungbox(z, lags=[ljungbox_lags], return_df=True)["lb_pvalue"].iloc[0]
                lb_z2 = acorr_ljungbox(z**2, lags=[ljungbox_lags], return_df=True)["lb_pvalue"].iloc[0]
                lb_absz = acorr_ljungbox(z.abs(), lags=[ljungbox_lags], return_df=True)["lb_pvalue"].iloc[0]

                model_name = spec.name()

                # persistence (only meaningful for GARCH(1,1)-ish; keep generic)
                alpha1 = res.params.get("alpha[1]", np.nan)
                beta1 = res.params.get("beta[1]", np.nan)
                persistence = alpha1 + beta1 if np.isfinite(alpha1) and np.isfinite(beta1) else np.nan

                rows.append(
                    {
                        "Asset": asset_name,
                        "Model": model_name,
                        "Dist": dist,
                        "AIC": res.aic,
                        "BIC": res.bic,
                        "LB(z) pval": float(lb_z),
                        "LB(|z|) pval": float(lb_absz),
                        "LB(z^2) pval": float(lb_z2),
                        "alpha+beta": float(persistence) if np.isfinite(persistence) else np.nan,
                    }
                )

                fitted_models[(model_name, dist)] = res

            except Exception as e:
                # keep it silent but you can log e if desired
                continue

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise RuntimeError(f"{asset_name}: no models successfully fit (check data / arch install)")

    results_df = results_df.sort_values(["AIC", "BIC"], ascending=True).reset_index(drop=True)
    return results_df, fitted_models




























def fit_vol_models_many_assets(
    returns_df: pd.DataFrame,
    *,
    model_grid: Iterable[VolModelSpec] = DEFAULT_MODEL_GRID,
    dists: Iterable[str] = DEFAULT_DISTS,
    mean: str = "Constant",
    ar_lags: int = 0,
    ljungbox_lags: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Dict[Tuple[str, str], object]]]:
    """
    Fit vol-model grids for each column in returns_df.

    Returns:
      - concatenated results table for all assets
      - nested dict: fitted[asset][(model_name, dist)] -> arch result
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df is empty")

    all_rows = []
    fitted_all: Dict[str, Dict[Tuple[str, str], object]] = {}

    for col in returns_df.columns:
        asset = str(col)
        res_table, fitted = fit_vol_models_one_asset(
            returns_df[col],
            asset_name=asset,
            model_grid=model_grid,
            dists=dists,
            mean=mean,
            ar_lags=ar_lags,
            ljungbox_lags=ljungbox_lags,
        )
        all_rows.append(res_table)
        fitted_all[asset] = fitted

    results_all = pd.concat(all_rows, axis=0, ignore_index=True)
    results_all = results_all.sort_values(["Asset", "AIC"], ascending=[True, True]).reset_index(drop=True)
    return results_all, fitted_all


















def select_best_models(
    results_all: pd.DataFrame,
    *,
    by: str = "AIC",
) -> pd.DataFrame:
    """
    For each asset, pick the row with minimum `by` (AIC by default).
    Returns a DataFrame indexed by Asset with the best model spec.
    """
    if results_all is None or results_all.empty:
        raise ValueError("results_all is empty")

    idx = results_all.groupby("Asset")[by].idxmin()
    best = results_all.loc[idx].copy().set_index("Asset")
    return best.sort_index()











def extract_vol_and_std_resid(
    fitted_models_by_asset: Dict[str, Dict[Tuple[str, str], object]],
    best_models: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Given fitted models and a best-model table, return:
      - vol_by_asset[ticker] = conditional volatility sigma_t (pd.Series)
      - z_by_asset[ticker] = standardized residuals z_t (pd.Series)
    """
    vol_by_asset: Dict[str, pd.Series] = {}
    z_by_asset: Dict[str, pd.Series] = {}

    for asset, row in best_models.iterrows():
        model_name = row["Model"]
        dist = row["Dist"]
        res = fitted_models_by_asset[asset][(model_name, dist)]

        # arch returns pandas Series aligned to original index usually
        sigma = pd.Series(res.conditional_volatility).dropna()
        z = pd.Series(res.std_resid).dropna()

        vol_by_asset[asset] = sigma
        z_by_asset[asset] = z

    return vol_by_asset, z_by_asset

















@dataclass
class CorrMatrices:
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    kendall: pd.DataFrame


def compute_corr_matrices(
    returns_df: pd.DataFrame,
    *,
    method_pearson: str = "pearson",
    method_spearman: str = "spearman",
    method_kendall: str = "kendall",
    min_periods: Optional[int] = None,
) -> CorrMatrices:
    """
    Compute Pearson, Spearman, and Kendall correlation matrices from a returns DataFrame.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Columns = assets, index = dates.
    min_periods : int or None
        Minimum overlapping observations required per pair.

    Returns
    -------
    CorrMatrices dataclass with pearson, spearman, kendall matrices.
    """
    if returns_df is None or returns_df.empty:
        raise ValueError("returns_df is empty")

    df = returns_df.copy()

    pearson = df.corr(method=method_pearson, min_periods=min_periods)
    spearman = df.corr(method=method_spearman, min_periods=min_periods)
    kendall = df.corr(method=method_kendall, min_periods=min_periods)

    return CorrMatrices(pearson=pearson, spearman=spearman, kendall=kendall)
















def dict_series_to_panel(
    series_by_asset: Dict[str, pd.Series],
    *,
    join: str = "inner",
    start: Optional[str] = None,
    end: Optional[str] = None,
    sort_index: bool = True,
) -> pd.DataFrame:
    """
    Convert {asset -> pd.Series} into a DataFrame (columns=assets), aligned by index.

    join:
      - "inner": keep only dates present for all assets
      - "outer": union of dates
    """
    if not series_by_asset:
        raise ValueError("series_by_asset is empty")

    cols = []
    for k, s in series_by_asset.items():
        if s is None:
            continue
        ss = pd.Series(s).dropna().copy()
        if not isinstance(ss.index, pd.DatetimeIndex):
            ss.index = pd.to_datetime(ss.index)
        if sort_index:
            ss = ss.sort_index()
        ss.name = str(k)
        cols.append(ss)

    if not cols:
        raise ValueError("No valid series found in series_by_asset")

    df = pd.concat(cols, axis=1, join=join)

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    if sort_index:
        df = df.sort_index()

    return df









def build_corr_panel_relative_to_current(
    z_panel: pd.DataFrame,
    portfolios: dict,
    current_name: str = "Current",
):
    """
    Build z_panel sliced to assets appearing in Current + any alternative portfolios.
    """
    current_assets = set(portfolios[current_name].keys())
    other_assets = set().union(*[w.keys() for w in portfolios.values()])
    assets = sorted(current_assets | other_assets)

    return z_panel[assets].dropna(how="all")

