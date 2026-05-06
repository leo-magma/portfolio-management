(() => {
  function init() {
    if (!window.flatpickr) return;

    const common = {
      dateFormat: "Y-m-d",
      allowInput: true,
      disableMobile: true, // force consistent picker UI
    };

    const start = document.querySelector("#id_start");
    const end = document.querySelector("#id_end");

    if (start) {
      window.flatpickr(start, {
        ...common,
        defaultDate: start.value || null,
      });
    }
    if (end) {
      window.flatpickr(end, {
        ...common,
        defaultDate: end.value || null,
      });
    }
  }

  window.addEventListener("DOMContentLoaded", init);
})();

