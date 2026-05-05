from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _base_layout(title: str, height: int = 420):
    return dict(
        title=title,
        template="plotly_dark",
        height=height,
        margin=dict(l=30, r=20, t=50, b=30),
        legend_title="Ticker",
    )


def plot_wealth(wealth_df: pd.DataFrame, title: str = "Wealth (from initial capital)") -> go.Figure:
    fig = go.Figure()
    for col in wealth_df.columns:
        s = wealth_df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text="Wealth")
    fig.update_xaxes(title_text="Date", tickformat="%Y-%m-%d")
    return fig


def plot_returns(rets_df: pd.DataFrame, title: str = "Returns (comparison)") -> go.Figure:
    """
    Auto visualization:
    - If many observations (e.g. FX daily), use line (Scattergl) for readability.
    - Otherwise, use grouped bars.
    """
    fig = go.Figure()
    n = int(len(rets_df.index))
    use_lines = n >= 90

    for col in rets_df.columns:
        s = rets_df[col].dropna()
        if s.empty:
            continue
        if use_lines:
            fig.add_trace(
                go.Scattergl(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name=str(col),
                    line=dict(width=1.4),
                    opacity=0.85,
                    hovertemplate="Ticker=%{fullData.name}<br>Date=%{x|%Y-%m-%d}<br>Return=%{y:.3%}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=s.index,
                    y=s.values,
                    name=str(col),
                    opacity=0.6,
                    hovertemplate="Ticker=%{fullData.name}<br>Date=%{x|%Y-%m-%d}<br>Return=%{y:.3%}<extra></extra>",
                )
            )

    fig.update_layout(**_base_layout(title), barmode="group" if not use_lines else None)
    fig.update_yaxes(title_text="Return", tickformat=".1%")
    fig.update_xaxes(title_text="Date", tickformat="%Y-%m-%d")
    return fig


def plot_drawdown(dd_df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    fig = go.Figure()
    for col in dd_df.columns:
        s = dd_df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
    fig.update_xaxes(title_text="Date", tickformat="%Y-%m-%d")
    return fig


def plot_corr_heatmap(corr: pd.DataFrame, title: str = "Correlation (returns)") -> go.Figure:
    if corr.empty:
        return go.Figure(layout=_base_layout(title))
    z = corr.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=corr.columns.astype(str),
            y=corr.index.astype(str),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="corr"),
        )
    )
    fig.update_layout(**_base_layout(title, height=520))
    return fig


def plot_rolling_line(df: pd.DataFrame, title: str, yaxis_title: str) -> go.Figure:
    fig = go.Figure()
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
    fig.update_layout(**_base_layout(title))
    fig.update_yaxes(title_text=yaxis_title)
    fig.update_xaxes(title_text="Date")
    return fig


def plot_return_distribution(rets_df: pd.DataFrame, title: str = "Return distribution") -> go.Figure:
    fig = go.Figure()
    for col in rets_df.columns:
        s = rets_df[col].dropna()
        if len(s) < 2:
            continue
        fig.add_trace(go.Histogram(x=s.values, name=str(col), opacity=0.55, nbinsx=40))
    fig.update_layout(**_base_layout(title, height=460), barmode="overlay")
    fig.update_xaxes(title_text="Return")
    fig.update_yaxes(title_text="Count")
    return fig


def plot_risk_return_scatter(
    metrics_rows: list[dict],
    title: str = "Risk–Return (CAGR vs Vol)",
) -> go.Figure:
    """
    x: annualized vol, y: CAGR. Color: Sharpe. Size: |MaxDD|.
    metrics_rows should contain: ticker, cagr, vol_annual, sharpe, max_drawdown.
    """
    xs = []
    ys = []
    colors = []
    sizes = []
    labels = []

    for r in metrics_rows:
        if r.get("cagr") is None or r.get("vol_annual") is None:
            continue
        xs.append(float(r["vol_annual"]))
        ys.append(float(r["cagr"]))
        colors.append(float(r["sharpe"]) if r.get("sharpe") is not None else 0.0)
        mdd = r.get("max_drawdown")
        sizes.append(10.0 + 35.0 * float(abs(mdd)) if mdd is not None else 12.0)
        labels.append(str(r.get("ticker")))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                line=dict(width=1, color="rgba(255,255,255,.35)"),
                opacity=0.9,
            ),
            hovertemplate="Ticker=%{text}<br>Vol=%{x:.4f}<br>CAGR=%{y:.4f}<extra></extra>",
            name="",
        )
    )
    fig.update_layout(**_base_layout(title, height=520))
    fig.update_xaxes(title_text="Volatility (annualized)")
    fig.update_yaxes(title_text="CAGR", tickformat=".1%")
    return fig


def plot_var_es_by_method(
    metrics_rows: list[dict],
    which: str = "var",
    title: str | None = None,
) -> go.Figure:
    """
    Grouped bar chart: each method is a series, x=ticker, y=VaR or ES.
    metrics_rows should include var_details (list of RiskResult).
    """
    which = which.lower()
    if which not in {"var", "es"}:
        raise ValueError("which must be 'var' or 'es'")

    by_method: dict[str, dict[str, float]] = {}
    tickers: list[str] = []

    for r in metrics_rows:
        t = str(r.get("ticker"))
        tickers.append(t)
        for rr in r.get("var_details") or []:
            m = getattr(rr, "method", None) or rr.get("method")  # dataclass or dict
            if not m:
                continue
            val = getattr(rr, which, None) if hasattr(rr, which) else rr.get(which)
            if val is None:
                continue
            by_method.setdefault(str(m), {})[t] = float(val)

    tickers = list(dict.fromkeys(tickers))  # de-dup preserve order
    fig = go.Figure()

    label_map = {
        "historical": "Historical",
        "delta_normal": "Delta-Normal",
        "monte_carlo": "Monte Carlo",
    }

    for method, series in by_method.items():
        ys = [series.get(t) for t in tickers]
        fig.add_trace(go.Bar(x=tickers, y=ys, name=label_map.get(method, method)))

    if title is None:
        title = "VaR by method" if which == "var" else "ES by method"
    fig.update_layout(**_base_layout(title, height=480), barmode="group")
    fig.update_yaxes(title_text=which.upper(), tickformat=".1%")
    fig.update_xaxes(title_text="Ticker")
    return fig


def plot_beta_alpha_scatter(
    metrics_rows: list[dict],
    title: str = "Benchmark exposure (beta vs alpha)",
) -> go.Figure:
    xs = []
    ys = []
    texts = []
    sizes = []
    colors = []
    for r in metrics_rows:
        b = r.get("beta")
        a = r.get("alpha_annual")
        if b is None or a is None:
            continue
        xs.append(float(b))
        ys.append(float(a))
        texts.append(str(r.get("ticker")))
        te = r.get("tracking_error_annual")
        sizes.append(10.0 + 25.0 * float(te) if te is not None else 12.0)
        ir = r.get("information_ratio")
        colors.append(float(ir) if ir is not None else 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=texts,
            textposition="top center",
            marker=dict(
                size=sizes,
                color=colors,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Info Ratio"),
                line=dict(width=1, color="rgba(255,255,255,.35)"),
                opacity=0.9,
            ),
            hovertemplate="Ticker=%{text}<br>beta=%{x:.3f}<br>alpha(ann.)=%{y:.3%}<extra></extra>",
            name="",
        )
    )
    fig.update_layout(**_base_layout(title, height=520))
    fig.update_xaxes(title_text="Beta")
    fig.update_yaxes(title_text="Alpha (annualized)", tickformat=".1%")
    return fig


def plot_attribution_stacked(
    contrib_df: pd.DataFrame,
    title: str = "Attribution (period contribution)",
) -> go.Figure:
    """
    contrib_df index: dates, columns: tickers, values: contribution to portfolio return.
    """
    fig = go.Figure()
    for col in contrib_df.columns:
        s = contrib_df[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col), stackgroup="one"))
    fig.update_layout(**_base_layout(title, height=520))
    fig.update_yaxes(title_text="Contribution to return")
    fig.update_xaxes(title_text="Date")
    return fig


def plot_attribution_cumulative(
    contrib_df: pd.DataFrame,
    title: str = "Attribution (cumulative contribution)",
) -> go.Figure:
    cum = contrib_df.fillna(0.0).cumsum()
    fig = go.Figure()
    for col in cum.columns:
        s = cum[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=str(col)))
    fig.update_layout(**_base_layout(title, height=520))
    fig.update_yaxes(title_text="Cumulative contribution")
    fig.update_xaxes(title_text="Date")
    return fig

