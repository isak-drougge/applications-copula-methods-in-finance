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

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Clear index/column "names" that show up as that annoying header
    df.index.name = None
    df.columns.name = None

    return df





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
    loader= None,
    show_progress: bool = True,
    drop_empty: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load price data for multiple tickers into a dict[ticker] = dataframe.

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

    return raw_prices
