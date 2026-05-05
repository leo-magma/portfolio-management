(() => {
  const presets = {
    equity: { tickers: "AAPL, MSFT, NVDA, AMZN", benchmark: "^GSPC", frequency: "1wk" },
    index: { tickers: "^GSPC, ^NDX, ^DJI", benchmark: "^GSPC", frequency: "1wk" },
    fx: { tickers: "EURUSD=X, JPY=X, GBPUSD=X", benchmark: "", frequency: "1d" },
  };

  function setValue(name, value) {
    const el = document.querySelector(`[name="${name}"]`);
    if (!el) return;
    el.value = value;
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function applyPreset(key) {
    const p = presets[key];
    if (!p) return;
    setValue("tickers", p.tickers);
    setValue("benchmark", p.benchmark);
    setValue("frequency", p.frequency);
  }

  function init() {
    document.querySelectorAll("[data-preset]").forEach((btn) => {
      btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
    });
    const quick = document.querySelector("[data-quick-run]");
    if (quick) {
      quick.addEventListener("click", () => {
        const form = document.querySelector("form");
        if (form) form.requestSubmit();
      });
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();

