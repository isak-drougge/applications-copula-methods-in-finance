from __future__ import annotations

from pathlib import Path
import pandas as pd
import yfinance as yf
from typing import Dict, Iterable




# Where data is stored locally (outside src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)




def _clean_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # If multiindex, take the first level (your existing convention)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalize index: datetime, sorted, unique, tz-naive
    idx = pd.to_datetime(df.index)
    # Make tz-naive if tz-aware
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    df.index = idx

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Clear index/column "names" that show up as that annoying header
    df.index.name = None
    df.columns.name = None

    return df





def _to_date_index(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Convert any datetime-like index to a tz-naive daily date index (midnight).
    This is the core invariant used to align FX and asset series robustly.
    """
    dti = pd.to_datetime(idx)
    if getattr(dti, "tz", None) is not None:
        dti = dti.tz_convert(None)
    return dti.normalize()





def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{ticker}_{start}_{end}.csv"

    # Try loading from cache
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = _clean_yfinance_columns(df)

            # sanity check: index must be datetime-like
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                raise ValueError("Cached file has non-datetime index")

            return df

        except Exception as e:
            print(f"[load_prices] Cache file is invalid, redownloading: {file_path.name}")
            # optionally: print(e)

    # Download fresh
    df = yf.download(ticker, start=start, end=end, progress=False, group_by="column")
    df = _clean_yfinance_columns(df)

    # Save clean CSV
    df.to_csv(file_path)
    return df












def load_prices_many(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    loader=None,
    show_progress: bool = True,
    drop_empty: bool = True,
    # --- NEW: currency handling ---
    base_ccy: str | None = None,
    fx_fill_method: str = "ffill",
    _currency_map: Dict[str, str] | None = None,   # optional override/cache
) -> Dict[str, pd.DataFrame]:
    """
    Load price data for multiple tickers into a dict[ticker] = dataframe.

    If base_ccy is provided, converts each ticker's prices into base_ccy using Yahoo FX rates.
    This is the industry-standard approach for multi-currency portfolio analysis.

    Parameters
    ----------
    tickers : iterable of str
        E.g. ["AAPL", "MSFT", ...]
    start, end : str
        Date strings like "2015-01-01".
    loader : callable
        Your single-ticker function, e.g. load_prices(ticker, start, end) -> pd.DataFrame.
        If None, uses `load_prices` from the current namespace.
    show_progress : bool
        Prints status lines per ticker.
    drop_empty : bool
        If True, silently drops tickers that return empty frames.
    base_ccy : str | None
        If provided (e.g. "SEK"), converts all price columns into this base currency.
    fx_fill_method : str
        How to align FX series to asset dates: "ffill" (standard), "bfill", or None.
    _currency_map : dict[str,str] | None
        Optional precomputed {ticker: currency} to avoid repeated Yahoo metadata calls.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    if loader is None:
        # Assumes you already have load_prices imported in your notebook/session
        loader = load_prices  # noqa: F821

    raw_prices: Dict[str, pd.DataFrame] = {}

    tickers = list(tickers)
    for i, t in enumerate(tickers, start=1):
        t = str(t).strip().upper()
        if show_progress:
            print(f"[{i}/{len(tickers)}] Loading {t}...")

        try:
            df = loader(t, start, end)
        except Exception as e:
            print(f"  -> FAILED for {t}: {type(e).__name__}: {e}")
            continue

        if df is None or (hasattr(df, "empty") and df.empty):
            msg = "  -> empty result"
            if drop_empty:
                if show_progress:
                    print(msg + " (dropped)")
                continue
            else:
                if show_progress:
                    print(msg + " (kept)")
                raw_prices[t] = df
                continue

        # Normalize index just in case
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        raw_prices[t] = df
        if show_progress:
            print(f"  -> ok: {df.shape[0]} rows, {df.shape[1]} cols")

    # --- NEW: optional FX conversion step ---
    if base_ccy is not None:
        # Import here to avoid import cost if not used
        # Assumes these helpers live in the same module or are importable here.
        # If they are in this file, remove these imports.
        try:
            from .yfinance_data_loader import get_ticker_currency_map, convert_prices_dict_to_base_currency  # type: ignore
        except Exception:
            # Fallback for same-file definitions
            try:
                get_ticker_currency_map  # noqa: F821
                convert_prices_dict_to_base_currency  # noqa: F821
            except NameError as e:
                raise NameError(
                    "Currency conversion requested (base_ccy not None) but helpers are missing. "
                    "Define get_ticker_currency_map and convert_prices_dict_to_base_currency in this module."
                ) from e

        used_tickers = list(raw_prices.keys())
        ccy_map = _currency_map or get_ticker_currency_map(used_tickers)

        if show_progress:
            # quick summary
            vc = pd.Series(ccy_map).value_counts(dropna=False)
            print(f"[FX] Converting to base currency: {base_ccy}")
            print(f"[FX] Currency counts:\n{vc}")

        raw_prices = convert_prices_dict_to_base_currency(
            raw_prices=raw_prices,
            ticker_ccy=ccy_map,
            base_ccy=base_ccy,
            start=start,
            end=end,
            fill_method=fx_fill_method,
        )

        if show_progress:
            print("[FX] Conversion complete.")

    return raw_prices











def _get_price_col(df: pd.DataFrame) -> str:
    for c in ("Adj Close", "Close"):
        if c in df.columns:
            return c
    raise ValueError(f"No 'Adj Close' or 'Close' column found. Columns={list(df.columns)}")




def get_ticker_currency_map(tickers: list[str]) -> dict[str, str]:
    """
    Returns {ticker: currency_code} using Yahoo fast_info (preferred).
    """
    out: dict[str, str] = {}
    for t in tickers:
        tt = str(t).strip().upper()
        fi = yf.Ticker(tt).fast_info
        ccy = fi.get("currency") or fi.get("priceCurrency")
        out[tt] = ccy
    return out



def fx_ticker(from_ccy: str, to_ccy: str) -> str:
    """
    Yahoo FX tickers: 'USDSEK=X' means 1 USD in SEK.
    """
    if from_ccy == to_ccy:
        return ""
    return f"{from_ccy}{to_ccy}=X"



def load_fx_series(from_ccy: str, to_ccy: str, start: str, end: str) -> pd.Series:
    """
    Download FX series as a daily tz-naive date-indexed Series.
    Value is: 1 unit of from_ccy in units of to_ccy.
    """
    tkr = fx_ticker(from_ccy, to_ccy)
    if not tkr:
        raise ValueError("from_ccy == to_ccy; no FX needed")

    fx = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=False)
    if fx is None or fx.empty:
        raise ValueError(f"Could not download FX series for {tkr}")

    col = _get_price_col(fx)
    s = fx[col].astype(float).copy()

    # make a robust daily index
    s.index = _to_date_index(s.index)
    s = s[~s.index.duplicated(keep="last")]  # de-dup by date
    s.name = tkr

    # sanity: avoid all-nan FX series
    if s.dropna().empty:
        raise ValueError(f"FX series {tkr} contains no non-NaN values.")

    return s



def convert_prices_dict_to_base_currency(
    raw_prices: dict[str, pd.DataFrame],
    ticker_ccy: dict[str, str],
    base_ccy: str,
    start: str,
    end: str,
    fill_method: str = "ffill",
) -> dict[str, pd.DataFrame]:
    """
    Convert each ticker's OHLC/Adj Close columns into base_ccy using Yahoo FX rates.

    Robust approach:
      - Convert both asset and FX to tz-naive date stamps
      - Align FX to asset dates using merge_asof (as-of join), which is standard for market data
      - Multiply price-like columns by the aligned FX

    This avoids catastrophic all-NaN conversion when indices differ in timezone/time-of-day/calendar.
    """

    needed_ccys = sorted({ccy for ccy in ticker_ccy.values() if ccy and ccy != base_ccy})

    fx_cache: dict[str, pd.Series] = {}
    for ccy in needed_ccys:
        fx_cache[ccy] = load_fx_series(ccy, base_ccy, start, end)  # should return daily date-indexed Series

    price_cols = {"Open", "High", "Low", "Close", "Adj Close"}

    def _cols_to_convert(df: pd.DataFrame) -> list:
        if isinstance(df.columns, pd.MultiIndex):
            return [col for col in df.columns if any(str(level) in price_cols for level in col)]
        return [col for col in df.columns if str(col) in price_cols]

    def _to_date_index(idx: pd.Index) -> pd.DatetimeIndex:
        dti = pd.to_datetime(idx)
        if getattr(dti, "tz", None) is not None:
            dti = dti.tz_convert(None)
        return dti.normalize()

    out: dict[str, pd.DataFrame] = {}

    for tkr, df in raw_prices.items():
        if df is None or getattr(df, "empty", False):
            out[tkr] = df
            continue

        ccy = ticker_ccy.get(tkr)
        if (not ccy) or (ccy == base_ccy):
            out[tkr] = df.copy()
            continue

        if ccy not in fx_cache:
            raise ValueError(f"Missing FX cache for {ccy}->{base_ccy} (ticker {tkr})")

        fx = fx_cache[ccy].copy()
        if fx.dropna().empty:
            raise ValueError(f"FX series for {ccy}->{base_ccy} is empty after download.")

        df2 = df.copy()
        cols = _cols_to_convert(df2)
        if not cols:
            out[tkr] = df2
            continue

        # Build date keys
        asset_dates = _to_date_index(df2.index)

        # FX already date-indexed, but enforce invariants
        fx.index = _to_date_index(fx.index)
        fx = fx[~fx.index.duplicated(keep="last")].sort_index()

        # As-of join: for each asset date, take the last FX observation <= that date
        # This is robust to gaps and differing calendars.
        fx_df = fx.reset_index()
        fx_df.columns = ["date", "fx"]

        asset_df = pd.DataFrame({"date": asset_dates})
        asset_df = asset_df.sort_values("date")

        merged = pd.merge_asof(
            asset_df,
            fx_df.sort_values("date"),
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )

        aligned_fx = pd.Series(merged["fx"].values, index=df2.index)

        # Optional fill for any remaining NaNs (e.g., asset starts before FX history)
        if fill_method == "ffill":
            aligned_fx = aligned_fx.ffill()
        elif fill_method == "bfill":
            aligned_fx = aligned_fx.bfill()
        elif fill_method is None:
            pass
        else:
            raise ValueError("fill_method must be 'ffill', 'bfill', or None")

        # If still all NaN: hard fail with diagnostics (this should not happen under normal conditions)
        if aligned_fx.dropna().empty:
            raise ValueError(
                f"FX alignment produced all-NaN FX for {tkr} ({ccy}->{base_ccy}). "
                f"Asset date range: [{asset_dates.min()} .. {asset_dates.max()}], "
                f"FX date range: [{fx.index.min()} .. {fx.index.max()}]."
            )

        # Multiply robustly (handles duplicate labels)
        df2.loc[:, cols] = df2.loc[:, cols].mul(aligned_fx, axis=0)

        out[tkr] = df2

    return out
