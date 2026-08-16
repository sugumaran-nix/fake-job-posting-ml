(() => {
  const table = document.getElementById('tbl-desktop');
  const cards = document.getElementById('cards-mobile');
  const layout = () => {
    if (!table || !cards) return;
    const desktop = window.innerWidth >= 640;
    table.style.display = desktop ? 'block' : 'none';
    cards.style.display = desktop ? 'none' : 'block';
  };
  layout();
  window.addEventListener('resize', layout, { passive: true });
  document.querySelectorAll('.stagger').forEach((element) => element.classList.add('go'));
})();
