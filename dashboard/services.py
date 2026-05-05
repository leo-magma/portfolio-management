from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .analytics import (
    RiskResult,
    ReturnType,
    VarMethod,
    annualized_vol,
    cagr_from_returns,
    drawdown_from_wealth,
    max_drawdown,
    resample_prices,
    returns_from_prices,
    sharpe_ratio,
    sortino_ratio,
    var_es,
    wealth_from_returns,
)

@dataclass(frozen=True)
class SeriesMetrics:
    ticker: str
    n_periods: int
    periods_per_year: float
    cagr: float | None
    vol_annual: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    beta: float | None
    alpha_annual: float | None
    tracking_error_annual: float | None
    information_ratio: float | None
    var_monthly: float | None
    es_monthly: float | None
    var_details: list[RiskResult]


def download_adj_close(
    tickers: list[str],
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # yfinance returns:
    # - MultiIndex columns for multiple tickers: ('Adj Close', 'AAPL'), ...
    # - single-level for one ticker: 'Adj Close'
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" not in df.columns.get_level_values(0)) and ("Close" in df.columns.get_level_values(0)):
            px = df["Close"].copy()
        else:
            px = df["Adj Close"].copy()
        px = px.rename_axis(None, axis=1)
        return px

    if "Adj Close" in df.columns:
        return df[["Adj Close"]].rename(columns={"Adj Close": tickers[0] if tickers else "price"})
    if "Close" in df.columns:
        return df[["Close"]].rename(columns={"Close": tickers[0] if tickers else "price"})
    return pd.DataFrame()


def _periods_per_year(freq: str) -> float:
    if freq == "1d":
        return 252.0
    if freq == "1wk":
        return 52.0
    if freq == "1mo":
        return 12.0
    return 12.0


def compute_panels(
    adj_close_daily: pd.DataFrame,
    initial_capital: float,
    confidence: float,
    rf_annual: float,
    frequency: str = "1mo",
    return_type: ReturnType = "simple",
    var_methods: list[VarMethod] | None = None,
    horizon: int = 1,
    mc_sims: int = 20000,
    benchmark_adj_close_daily: pd.Series | None = None,
    rolling_window: int | None = None,
) -> dict[str, Any]:
    if adj_close_daily.empty:
        return {
            "tickers": [],
            "returns": pd.DataFrame(),
            "wealth": pd.DataFrame(),
            "drawdown": pd.DataFrame(),
            "metrics": [],
            "corr": pd.DataFrame(),
            "benchmark": None,
            "rolling": {},
        }

    freq = frequency
    ppy = _periods_per_year(freq)
    var_methods = var_methods or ["historical"]

    benchmark = None
    benchmark_wealth = None
    benchmark_prices = None
    if benchmark_adj_close_daily is not None and not benchmark_adj_close_daily.dropna().empty:
        bpx = resample_prices(benchmark_adj_close_daily, freq=freq)  # type: ignore[arg-type]
        benchmark_prices = bpx
        benchmark = returns_from_prices(bpx, return_type=return_type)
        benchmark.name = "Benchmark"
        if not bpx.dropna().empty:
            benchmark_wealth = (bpx / float(bpx.iloc[0])) * float(initial_capital)
            benchmark_wealth.name = "Benchmark"

    returns: dict[str, pd.Series] = {}
    wealth: dict[str, pd.Series] = {}
    drawdown: dict[str, pd.Series] = {}
    metrics: list[SeriesMetrics] = []
    prices_resampled: dict[str, pd.Series] = {}

    for ticker in adj_close_daily.columns:
        px = resample_prices(adj_close_daily[ticker], freq=freq)  # type: ignore[arg-type]
        prices_resampled[str(ticker)] = px
        rets = returns_from_prices(px, return_type=return_type)
        rets.name = str(ticker)
        returns[ticker] = rets
        if rets.empty:
            wealth[ticker] = pd.Series(dtype=float, name=ticker)
            drawdown[ticker] = pd.Series(dtype=float, name=ticker)
            metrics.append(
                SeriesMetrics(
                    ticker=ticker,
                    n_periods=0,
                    periods_per_year=ppy,
                    cagr=None,
                    vol_annual=None,
                    sharpe=None,
                    sortino=None,
                    max_drawdown=None,
                    beta=None,
                    alpha_annual=None,
                    tracking_error_annual=None,
                    information_ratio=None,
                    var_monthly=None,
                    es_monthly=None,
                    var_details=[],
                )
            )
            continue

        # Wealth should visibly start at initial capital: use price index ratio.
        w = (px / float(px.iloc[0])) * float(initial_capital)
        w.name = str(ticker)
        wealth[ticker] = w
        dd = drawdown_from_wealth(w)
        dd.name = str(ticker)
        drawdown[ticker] = dd

        risks = [var_es(rets, method=m, confidence=confidence, horizon=horizon, mc_sims=mc_sims) for m in var_methods]
        # keep backward compatible headline fields (first method)
        first = risks[0] if risks else None

        beta = None
        alpha_annual = None
        te_annual = None
        ir = None
        if benchmark is not None and not benchmark.empty:
            aligned = pd.concat([rets, benchmark], axis=1).dropna()
            if aligned.shape[0] >= 30:
                r_i = aligned.iloc[:, 0].astype(float)
                r_b = aligned.iloc[:, 1].astype(float)
                var_b = float(r_b.var(ddof=1))
                if var_b > 0.0:
                    cov = float(np.cov(r_i.values, r_b.values, ddof=1)[0, 1])
                    beta = cov / var_b
                    alpha = float(r_i.mean() - beta * r_b.mean())
                    alpha_annual = alpha * ppy

                active = r_i - r_b
                if active.shape[0] >= 2:
                    te_annual = float(active.std(ddof=1) * np.sqrt(ppy))
                    if te_annual and te_annual > 0.0:
                        ir = float((active.mean() * ppy) / te_annual)
        metrics.append(
            SeriesMetrics(
                ticker=ticker,
                n_periods=int(len(rets)),
                periods_per_year=ppy,
                cagr=cagr_from_returns(rets, periods_per_year=ppy),
                vol_annual=annualized_vol(rets, periods_per_year=ppy),
                sharpe=sharpe_ratio(rets, rf_annual=rf_annual, periods_per_year=ppy),
                sortino=sortino_ratio(rets, rf_annual=rf_annual, periods_per_year=ppy),
                max_drawdown=max_drawdown(dd),
                beta=beta,
                alpha_annual=alpha_annual,
                tracking_error_annual=te_annual,
                information_ratio=ir,
                var_monthly=None if first is None else first.var,
                es_monthly=None if first is None else first.es,
                var_details=risks,
            )
        )

    rets_df = pd.concat(returns.values(), axis=1).sort_index()
    wealth_df = pd.concat(wealth.values(), axis=1).sort_index()
    dd_df = pd.concat(drawdown.values(), axis=1).sort_index()
    corr = rets_df.corr() if not rets_df.empty else pd.DataFrame()

    # Equal-weight portfolio (uses overlapping periods only)
    portfolio_returns = None
    portfolio_wealth = None
    if prices_resampled:
        px_df = pd.concat(prices_resampled.values(), axis=1)
        px_df.columns = list(prices_resampled.keys())
        px_aligned = px_df.dropna(how="any")
        if not px_aligned.empty:
            normalized = px_aligned / px_aligned.iloc[0]
            portfolio_wealth = normalized.mean(axis=1) * float(initial_capital)
            portfolio_wealth.name = "Portfolio"
            portfolio_returns = portfolio_wealth.pct_change().dropna()
            portfolio_returns.name = "Portfolio"

    portfolio = {}
    if portfolio_wealth is not None and portfolio_returns is not None and not portfolio_returns.empty:
        p_dd = drawdown_from_wealth(portfolio_wealth)
        p_mdd = max_drawdown(p_dd)
        p_final = float(portfolio_wealth.iloc[-1])
        p_pnl = p_final - float(initial_capital)

        # Portfolio headline VaR/ES uses first selected method
        p_risks = [var_es(portfolio_returns, method=m, confidence=confidence, horizon=horizon, mc_sims=mc_sims) for m in var_methods]
        p_first = p_risks[0] if p_risks else None

        p_te = None
        if benchmark is not None and not benchmark.empty:
            ab = pd.concat([portfolio_returns, benchmark], axis=1).dropna()
            if ab.shape[0] >= 2:
                active = ab.iloc[:, 0] - ab.iloc[:, 1]
                p_te = float(active.std(ddof=1) * np.sqrt(ppy))

        portfolio = {
            "returns": portfolio_returns,
            "wealth": portfolio_wealth,
            "drawdown": p_dd,
            "pnl": p_pnl,
            "final": p_final,
            "max_drawdown": p_mdd,
            "var": None if p_first is None else p_first.var,
            "es": None if p_first is None else p_first.es,
            "tracking_error_annual": p_te,
            "risk_details": p_risks,
        }

    # Attribution for equal-weight portfolio (period contribution)
    attribution = {}
    if not rets_df.empty:
        aligned = rets_df.dropna(how="any")
        if not aligned.empty:
            n = aligned.shape[1]
            contrib = aligned / float(n)  # equal-weight contribution to portfolio return
            attribution = {
                "contrib": contrib,
                "contrib_cum": contrib.fillna(0.0).cumsum(),
            }

    rolling: dict[str, Any] = {}
    if rolling_window and rolling_window >= 2 and not rets_df.empty:
        roll_vol = rets_df.rolling(rolling_window).std() * np.sqrt(ppy)
        rolling["vol_annual"] = roll_vol

        rf_period = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0
        excess = rets_df - rf_period
        roll_mean = excess.rolling(rolling_window).mean()
        roll_std = excess.rolling(rolling_window).std()
        rolling["sharpe"] = (roll_mean / roll_std) * np.sqrt(ppy)

    return {
        "tickers": list(adj_close_daily.columns),
        "returns": rets_df,
        "wealth": wealth_df,
        "drawdown": dd_df,
        "metrics": metrics,
        "corr": corr,
        "benchmark": benchmark,
        "benchmark_wealth": benchmark_wealth,
        "benchmark_prices": benchmark_prices,
        "portfolio": portfolio,
        "attribution": attribution,
        "rolling": rolling,
    }

