from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

import plotly.graph_objects as go

from .forms import DashboardForm
from .plots import (
    plot_corr_heatmap,
    plot_drawdown,
    plot_beta_alpha_scatter,
    plot_attribution_cumulative,
    plot_attribution_stacked,
    plot_risk_return_scatter,
    plot_return_distribution,
    plot_returns,
    plot_var_es_by_method,
    plot_rolling_line,
    plot_wealth,
)
from .services import compute_panels, download_adj_close


def index(request: HttpRequest) -> HttpResponse:
    form = DashboardForm(request.GET or None)
    ctx = {"form": form}

    if not form.is_valid():
        return render(request, "dashboard/index.html", ctx)

    tickers = form.cleaned_data["tickers"]
    start = form.cleaned_data.get("start")
    end = form.cleaned_data.get("end")
    frequency = form.cleaned_data["frequency"]
    return_type = form.cleaned_data["return_type"]
    initial_capital = form.cleaned_data["initial_capital"]
    confidence = form.cleaned_data["confidence"]
    var_methods = form.cleaned_data["var_method"]
    horizon = form.cleaned_data["horizon"]
    mc_sims = form.cleaned_data["mc_sims"]
    benchmark = (form.cleaned_data.get("benchmark") or "").strip() or None
    rolling_window = form.cleaned_data.get("rolling_window")
    rf_annual = form.cleaned_data["rf_annual"]

    prices = download_adj_close(tickers=tickers, start=start, end=end)
    bench_px = None
    if benchmark:
        bdf = download_adj_close(tickers=[benchmark], start=start, end=end)
        if not bdf.empty:
            bench_px = bdf.iloc[:, 0]

    panels = compute_panels(
        adj_close_daily=prices,
        initial_capital=initial_capital,
        confidence=confidence,
        rf_annual=rf_annual,
        frequency=frequency,
        return_type=return_type,
        var_methods=var_methods,
        horizon=horizon,
        mc_sims=mc_sims,
        benchmark_adj_close_daily=bench_px,
        rolling_window=rolling_window,
    )

    wealth_df = panels["wealth"]
    rets_df = panels["returns"]
    dd_df = panels["drawdown"]
    corr = panels["corr"]
    rolling = panels["rolling"]
    metrics = panels["metrics"]
    portfolio = panels.get("portfolio") or {}
    attribution = panels.get("attribution") or {}
    benchmark_rets = panels.get("benchmark")
    benchmark_wealth = panels.get("benchmark_wealth")

    if wealth_df.empty:
        ctx["error"] = "No data returned. Please check tickers and date range."
        return render(request, "dashboard/index.html", ctx)

    wealth_fig = plot_wealth(wealth_df, title="Wealth (from initial capital)")
    rets_fig = plot_returns(rets_df, title="Returns (comparison)")
    dd_fig = plot_drawdown(dd_df, title="Drawdown")
    corr_fig = plot_corr_heatmap(corr, title="Correlation (returns)")
    dist_fig = plot_return_distribution(rets_df, title="Return distribution")
    scatter_fig = plot_risk_return_scatter([m.__dict__ for m in metrics], title="Risk–Return (CAGR vs Vol)")
    var_methods_fig = plot_var_es_by_method([m.__dict__ for m in metrics], which="var", title="VaR by method")
    es_methods_fig = plot_var_es_by_method([m.__dict__ for m in metrics], which="es", title="ES by method")
    bench_fig = plot_beta_alpha_scatter([m.__dict__ for m in metrics], title="Benchmark exposure (beta vs alpha)")

    # Overlay benchmark on wealth/returns when provided
    if benchmark_wealth is not None and not benchmark_wealth.dropna().empty:
        s = benchmark_wealth.dropna()
        wealth_fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=str(s.name or "Benchmark"),
                line=dict(dash="dot", width=2),
                opacity=0.9,
            )
        )

    if benchmark_rets is not None and not benchmark_rets.dropna().empty:
        s = benchmark_rets.dropna()
        rets_fig.add_trace(
            go.Bar(
                x=s.index,
                y=s.values,
                name=str(s.name or "Benchmark"),
                opacity=0.35,
            )
        )

    contrib = attribution.get("contrib")
    attrib_period_fig = None
    attrib_cum_fig = None
    if contrib is not None and not contrib.empty:
        attrib_period_fig = plot_attribution_stacked(contrib, title="Attribution (period contribution)")
        attrib_cum_fig = plot_attribution_cumulative(contrib, title="Attribution (cumulative contribution)")

    rolling_vol_fig = None
    rolling_sharpe_fig = None
    if isinstance(rolling, dict) and "vol_annual" in rolling and not rolling["vol_annual"].empty:
        rolling_vol_fig = plot_rolling_line(rolling["vol_annual"], title="Rolling volatility (annualized)", yaxis_title="Vol (ann.)")
    if isinstance(rolling, dict) and "sharpe" in rolling and not rolling["sharpe"].empty:
        rolling_sharpe_fig = plot_rolling_line(rolling["sharpe"], title="Rolling Sharpe", yaxis_title="Sharpe")

    ctx.update(
        {
            "tickers": panels["tickers"],
            "metrics": metrics,
            "portfolio": portfolio,
            "wealth_plot_html": wealth_fig.to_html(full_html=False, include_plotlyjs=False),
            "returns_plot_html": rets_fig.to_html(full_html=False, include_plotlyjs=False),
            "drawdown_plot_html": dd_fig.to_html(full_html=False, include_plotlyjs=False),
            "corr_plot_html": corr_fig.to_html(full_html=False, include_plotlyjs=False),
            "dist_plot_html": dist_fig.to_html(full_html=False, include_plotlyjs=False),
            "scatter_plot_html": scatter_fig.to_html(full_html=False, include_plotlyjs=False),
            "var_methods_plot_html": var_methods_fig.to_html(full_html=False, include_plotlyjs=False),
            "es_methods_plot_html": es_methods_fig.to_html(full_html=False, include_plotlyjs=False),
            "benchmark_plot_html": bench_fig.to_html(full_html=False, include_plotlyjs=False),
            "attrib_period_plot_html": None if attrib_period_fig is None else attrib_period_fig.to_html(full_html=False, include_plotlyjs=False),
            "attrib_cum_plot_html": None if attrib_cum_fig is None else attrib_cum_fig.to_html(full_html=False, include_plotlyjs=False),
            "rolling_vol_plot_html": None if rolling_vol_fig is None else rolling_vol_fig.to_html(full_html=False, include_plotlyjs=False),
            "rolling_sharpe_plot_html": None if rolling_sharpe_fig is None else rolling_sharpe_fig.to_html(full_html=False, include_plotlyjs=False),
            "last_updated": wealth_df.index.max(),
        }
    )
    return render(request, "dashboard/index.html", ctx)
