(() => {
  const table = document.getElementById('tbl-desktop');
  const cards = document.getElementById('cards-mobile');
  const searchInput = document.getElementById('history-search');
  const filterSelect = document.getElementById('history-filter');
  const status = document.getElementById('history-filter-status');
  const items = Array.from(document.querySelectorAll('[data-history-item]'));

  const layout = () => {
    if (!table || !cards) return;
    const desktop = window.innerWidth >= 640;
    table.style.display = desktop ? 'block' : 'none';
    cards.style.display = desktop ? 'none' : 'block';
  };

  const matchesVerdict = (item, filter) => {
    if (filter === 'all') return true;
    if (filter === 'fraud') return Boolean(item.querySelector('.pill-fraud'));
    return Boolean(item.querySelector('.pill-legit'));
  };

  const applyFilters = () => {
    const query = (searchInput?.value || '').trim().toLowerCase();
    const filter = filterSelect?.value || 'all';
    let visible = 0;
    items.forEach((item) => {
      const text = item.textContent.toLowerCase();
      const show = (!query || text.includes(query)) && matchesVerdict(item, filter);
      item.hidden = !show;
      if (show) visible += 1;
    });
    if (status) {
      const divisor = (table && cards) ? 2 : 1;
      const shown = Math.round(visible / divisor);
      const total = Math.round(items.length / divisor);
      status.textContent = shown === total ? `${total} shown` : `${shown} of ${total} shown`;
    }
  };

  document.querySelectorAll('[data-confirm-clear]').forEach((form) => {
    form.addEventListener('submit', (event) => {
      if (!window.confirm('Clear all history? This cannot be undone.')) event.preventDefault();
    });
  });
  document.querySelectorAll('[data-confirm-delete]').forEach((form) => {
    form.addEventListener('submit', (event) => {
      if (!window.confirm('Remove this analysis from History?')) event.preventDefault();
    });
  });

  searchInput?.addEventListener('input', applyFilters);
  filterSelect?.addEventListener('change', applyFilters);
  layout();
  applyFilters();
  window.addEventListener('resize', layout, { passive: true });
  document.querySelectorAll('.stagger').forEach((element) => element.classList.add('go'));
})();
