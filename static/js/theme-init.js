(() => {
  let theme = 'light';
  try {
    const stored = localStorage.getItem('jg-theme');
    if (stored === 'dark' || stored === 'light') {
      theme = stored;
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      theme = 'dark';
    }
  } catch (_) {
    // Keep the safe light default when storage is unavailable.
  }
  document.documentElement.setAttribute('data-theme', theme);
})();
