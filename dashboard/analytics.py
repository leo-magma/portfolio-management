from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

Freq = Literal["1d", "1wk", "1mo"]
ReturnType = Literal["simple", "log"]
VarMethod = Literal["historical", "delta_normal", "monte_carlo"]


@dataclass(frozen=True)
class RiskResult:
    method: VarMethod
    confidence: float
    horizon: int
    var: float | None  # positive loss number (e.g. 0.05 => -5%)
    es: float | None   # positive loss number


def resample_prices(px: pd.Series, freq: Freq) -> pd.Series:
    s = px.dropna()
    if s.empty:
        return s
    if freq == "1d":
        return s
    if freq == "1wk":
        return s.resample("W-FRI").last().dropna()
    if freq == "1mo":
        return s.resample("ME").last().dropna()
    raise ValueError(f"Unsupported freq: {freq}")


def returns_from_prices(px: pd.Series, return_type: ReturnType) -> pd.Series:
    s = px.dropna()
    if len(s) < 2:
        return pd.Series(dtype=float, name="ret")
    if return_type == "simple":
        out = s.pct_change().dropna()
    elif return_type == "log":
        out = np.log(s).diff().dropna()
    else:
        raise ValueError(f"Unsupported return_type: {return_type}")
    out.name = "ret"
    return out


def wealth_from_returns(rets: pd.Series, initial_capital: float) -> pd.Series:
    if rets.empty:
        return pd.Series(dtype=float, name="wealth")
    if rets.name is None:
        r = rets
    else:
        r = rets
    return (1.0 + r).cumprod() * float(initial_capital)


def drawdown_from_wealth(wealth: pd.Series) -> pd.Series:
    w = wealth.dropna()
    if w.empty:
        return pd.Series(dtype=float, name="drawdown")
    peak = w.cummax()
    dd = (w / peak) - 1.0
    dd.name = "drawdown"
    return dd


def cagr_from_returns(rets: pd.Series, periods_per_year: float) -> float | None:
    if rets.empty:
        return None
    growth = float(np.prod(1.0 + rets.values))
    years = len(rets) / periods_per_year
    if years <= 0:
        return None
    return growth ** (1.0 / years) - 1.0


def annualized_vol(rets: pd.Series, periods_per_year: float) -> float | None:
    if len(rets) < 2:
        return None
    return float(rets.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(rets: pd.Series, rf_annual: float, periods_per_year: float) -> float | None:
    if len(rets) < 2:
        return None
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = rets - rf_period
    vol = float(excess.std(ddof=1))
    if vol == 0.0:
        return None
    return float((excess.mean() / vol) * np.sqrt(periods_per_year))


def sortino_ratio(rets: pd.Series, rf_annual: float, periods_per_year: float) -> float | None:
    if len(rets) < 2:
        return None
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = rets - rf_period
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return None
    dd = float(downside.std(ddof=1))
    if dd == 0.0:
        return None
    return float((excess.mean() / dd) * np.sqrt(periods_per_year))


def max_drawdown(drawdown: pd.Series) -> float | None:
    dd = drawdown.dropna()
    if dd.empty:
        return None
    return float(dd.min())


def _var_es_historical(rets: pd.Series, confidence: float, horizon: int) -> tuple[float | None, float | None]:
    if len(rets) < 30:
        return None, None
    alpha = 1.0 - confidence
    horizon_rets = (1.0 + rets).rolling(horizon).apply(np.prod, raw=True) - 1.0 if horizon > 1 else rets
    horizon_rets = horizon_rets.dropna()
    if len(horizon_rets) < 30:
        return None, None
    q = float(np.quantile(horizon_rets.values, alpha))
    # If the left-tail quantile is positive (rare but possible), loss-VaR is 0.
    var = max(0.0, -q)
    tail = horizon_rets[horizon_rets <= q]
    es = None if tail.empty else -float(tail.mean())
    return var, es


def _var_es_delta_normal(rets: pd.Series, confidence: float, horizon: int) -> tuple[float | None, float | None]:
    if len(rets) < 30:
        return None, None
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))
    if sigma <= 0.0:
        return None, None

    # scale to horizon (assuming i.i.d. returns)
    mu_h = mu * horizon
    sigma_h = sigma * np.sqrt(horizon)

    alpha = 1.0 - confidence
    z = float(norm.ppf(alpha))  # negative
    q = mu_h + sigma_h * z      # return quantile (left tail)
    var = max(0.0, -q)

    # ES for Normal: E[R | R <= q] = mu - sigma * pdf(z)/alpha
    es_ret = mu_h - sigma_h * float(norm.pdf(z)) / alpha
    es = max(0.0, -es_ret)
    return var, es


def _var_es_monte_carlo(rets: pd.Series, confidence: float, horizon: int, sims: int, seed: int = 42) -> tuple[float | None, float | None]:
    if len(rets) < 30:
        return None, None
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=1))
    if sigma <= 0.0:
        return None, None

    rng = np.random.default_rng(seed)
    # simulate horizon simple returns with normal shocks and aggregate additively
    # (kept simple/fast; works best for small returns)
    paths = rng.normal(loc=mu, scale=sigma, size=(sims, horizon))
    sim_h = paths.sum(axis=1)

    alpha = 1.0 - confidence
    q = float(np.quantile(sim_h, alpha))
    var = max(0.0, -q)
    tail = sim_h[sim_h <= q]
    es = None if tail.size == 0 else -float(tail.mean())
    return var, es


def var_es(
    rets: pd.Series,
    method: VarMethod,
    confidence: float,
    horizon: int,
    mc_sims: int = 20000,
) -> RiskResult:
    if method == "historical":
        v, e = _var_es_historical(rets, confidence, horizon)
    elif method == "delta_normal":
        v, e = _var_es_delta_normal(rets, confidence, horizon)
    elif method == "monte_carlo":
        v, e = _var_es_monte_carlo(rets, confidence, horizon, sims=mc_sims)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return RiskResult(method=method, confidence=confidence, horizon=horizon, var=v, es=e)

