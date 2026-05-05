(() => {
  function initTabs(root) {
    const tabs = Array.from(root.querySelectorAll("[role='tab']"));
    const panels = Array.from(root.querySelectorAll("[role='tabpanel']"));
    if (tabs.length === 0 || panels.length === 0) return;

    function setActive(id) {
      for (const t of tabs) {
        t.setAttribute("aria-selected", t.dataset.target === id ? "true" : "false");
      }
      for (const p of panels) {
        p.dataset.active = p.id === id ? "true" : "false";
      }
    }

    for (const t of tabs) {
      t.addEventListener("click", () => setActive(t.dataset.target));
    }

    const first = tabs[0]?.dataset.target;
    if (first) setActive(first);
  }

  window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-tabs]").forEach(initTabs);
  });
})();

