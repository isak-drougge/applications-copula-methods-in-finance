from pathlib import Path
import pandas as pd
import yfinance as yf



# Where data is stored locally (outside src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)




def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load price data for a ticker.
    - Downloads once from Yahoo
    - Saves locally
    - Reloads from disk on future calls
    """
    file_path = DATA_DIR / f"{ticker}_{start}_{end}.csv"

    # Case 1: data already downloaded
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Case 2: first time → download
    df = yf.download(ticker, start=start, end=end)
    df.to_csv(file_path)
    return df
