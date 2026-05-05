from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from django import forms


def _split_tickers(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    tickers = [p for p in parts if p]
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


class DashboardForm(forms.Form):
    tickers = forms.CharField(
        label="Tickers (comma-separated)",
        help_text="e.g. AAPL, MSFT, 7203.T, ^N225",
        initial="AAPL,MSFT",
        widget=forms.Textarea(attrs={"rows": 2, "placeholder": "AAPL, MSFT, 7203.T"}),
    )
    start = forms.DateField(
        label="Start date",
        required=False,
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    end = forms.DateField(
        label="End date",
        required=False,
        widget=forms.DateInput(attrs={"type": "date"}),
    )
    frequency = forms.ChoiceField(
        label="Frequency",
        choices=[
            ("1d", "Daily"),
            ("1wk", "Weekly"),
            ("1mo", "Monthly"),
        ],
        initial="1mo",
    )
    return_type = forms.ChoiceField(
        label="Returns",
        choices=[
            ("simple", "Simple"),
            ("log", "Log"),
        ],
        initial="simple",
    )
    initial_capital = forms.FloatField(
        label="Initial capital",
        initial=10000,
        min_value=0,
        widget=forms.NumberInput(attrs={"step": "100"}),
    )
    confidence = forms.FloatField(
        label="VaR/ES confidence (e.g. 0.95)",
        initial=0.95,
        min_value=0.5,
        max_value=0.999,
        widget=forms.NumberInput(attrs={"step": "0.01"}),
    )
    var_method = forms.MultipleChoiceField(
        label="VaR/ES methods",
        required=False,
        choices=[
            ("historical", "Historical"),
            ("delta_normal", "Delta-Normal (Gaussian)"),
            ("monte_carlo", "Monte Carlo (Gaussian)"),
        ],
        initial=["historical", "delta_normal"],
        widget=forms.CheckboxSelectMultiple,
    )
    horizon = forms.IntegerField(
        label="Risk horizon (periods)",
        initial=1,
        min_value=1,
        max_value=60,
        help_text="If Frequency=Monthly, 1 means 1-month horizon. If Daily, 1 means 1-day horizon.",
    )
    mc_sims = forms.IntegerField(
        label="MC simulations",
        initial=20000,
        min_value=1000,
        max_value=200000,
    )
    benchmark = forms.CharField(
        label="Benchmark (optional)",
        required=False,
        help_text="e.g. ^GSPC, ^N225",
        widget=forms.TextInput(attrs={"placeholder": "^GSPC"}),
    )
    rolling_window = forms.IntegerField(
        label="Rolling window (optional)",
        required=False,
        min_value=6,
        max_value=120,
        help_text="e.g. 36 (rolling Sharpe/Vol over 36 periods)",
    )
    rf_annual = forms.FloatField(
        label="Risk-free rate (annual)",
        initial=0.0,
        widget=forms.NumberInput(attrs={"step": "0.001"}),
    )

    def clean_tickers(self) -> list[str]:
        raw = self.cleaned_data["tickers"]
        tickers = _split_tickers(raw)
        if not tickers:
            raise forms.ValidationError("Please input at least one ticker.")
        if len(tickers) > 12:
            raise forms.ValidationError("Up to 12 tickers per run.")
        return tickers

    def clean_var_method(self) -> list[str]:
        methods = self.cleaned_data.get("var_method") or []
        if not methods:
            return ["historical"]
        return methods

