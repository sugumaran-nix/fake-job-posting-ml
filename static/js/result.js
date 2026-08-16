(() => {
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.conf-fill[data-target]').forEach((element) => {
      window.setTimeout(() => { element.style.width = `${element.dataset.target}%`; }, 150);
    });
    document.querySelectorAll('.bar-fill[data-target]').forEach((element, index) => {
      window.setTimeout(() => { element.style.width = `${element.dataset.target}%`; }, 300 + index * 30);
    });
  });
})();
