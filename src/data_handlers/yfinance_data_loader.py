from pathlib import Path
import pandas as pd
import yfinance as yf



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

