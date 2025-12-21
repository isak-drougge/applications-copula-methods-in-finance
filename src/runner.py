from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_handlers.yfinance_data_loader import load_prices
from utils.visualization import plot_acf_scaled


@dataclass
class StepAResult:
    table: pd.DataFrame
    meta: dict


def run_stepA_joint_probabilities(
    tickers: list[str],
    start: str,
    end: str,
    thresholds: list[float],
    copulas: list[str] = ["gaussian", "t"],
    show_steps_in_plots: bool = True,
    acf_lags: int = 50,
    acf_conf_level: float = 0.95,
) -> StepAResult:

    def _maybe_pause(msg: str):
        if show_steps_in_plots:
            ans = input(f"{msg} Proceed? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                raise KeyboardInterrupt("Execution stopped by user.")

    # -------------------------
    # 1) Load and cache prices
    # -------------------------
    price_series = {}
    for t in tickers:
        df = load_prices(t, start=start, end=end)
        if "Close" not in df.columns:
            raise ValueError(f"'Close' column missing for ticker {t}")
        price_series[t] = df["Close"]

    prices = pd.DataFrame(price_series).dropna(how="all")

    if show_steps_in_plots:
        ax = prices.plot(figsize=(10, 4))
        ax.set_title(f"Prices (Close): {', '.join(tickers)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.tight_layout()
        plt.show()
        _maybe_pause("Prices plotted.")

    # -------------------------
    # 2) Compute log returns
    # -------------------------
    returns = np.log(prices).diff().dropna()

    if show_steps_in_plots:
        ax = returns.plot(figsize=(10, 4))
        ax.set_title(f"Log returns: {', '.join(tickers)}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log return")
        plt.tight_layout()
        plt.show()
        _maybe_pause("Returns plotted.")

    # -------------------------
    # 2.5) Heteroskedasticity quick-check: ACF(r) vs ACF(|r|)
    # -------------------------
    if show_steps_in_plots:
        n = len(tickers)
        fig, axes = plt.subplots(2, n, figsize=(6 * n, 7), squeeze=False)

        for j, t in enumerate(tickers):
            r = returns[t]

            plot_acf_scaled(
                r,
                axes[0, j],
                f"{t}: ACF of log returns",
                lags=acf_lags,
                conf_level=acf_conf_level,
            )
            plot_acf_scaled(
                r.abs(),
                axes[1, j],
                f"{t}: ACF of |log returns|",
                lags=acf_lags,
                conf_level=acf_conf_level,
            )

        plt.tight_layout()
        plt.show()
        _maybe_pause("ACF diagnostics plotted (returns vs |returns|).")

    return StepAResult(
        table=returns,
        meta={
            "tickers": tickers,
            "start": start,
            "end": end,
            "n_obs_prices": int(len(prices)),
            "n_obs_returns": int(len(returns)),
            "acf_lags": int(acf_lags),
            "acf_conf_level": float(acf_conf_level),
        },
    )


res = run_stepA_joint_probabilities(
    ["AAPL", "MSFT"],
    "2015-01-01",
    "2024-01-01",
    thresholds=[-0.03],
    show_steps_in_plots=True,
)
