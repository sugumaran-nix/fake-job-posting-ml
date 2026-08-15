'use strict';

/* ── Theme toggle ─────────────────────────────────────────────── */
(function () {
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('jg-theme', t);
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    const isDark = t === 'dark';
    btn.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    btn.setAttribute('aria-pressed', String(isDark));
    // update theme-color meta
    document.querySelectorAll('meta[name="theme-color"]').forEach(m => {
      const media = m.getAttribute('media') || '';
      if (media.includes('dark'))  m.setAttribute('content', isDark ? '#000000' : '#000000');
      if (media.includes('light')) m.setAttribute('content', isDark ? '#1C1C1E' : '#F2F2F7');
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;
    // Set initial aria-pressed state
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    btn.setAttribute('aria-pressed', String(current === 'dark'));
    btn.setAttribute('aria-label', current === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');

    btn.addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next);
    });
  });
})();

/* ── DOM ready ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {

  /* ── Nav hamburger ─────────────────────────────────────────── */
  const ham  = document.getElementById('hamburger');
  const menu = document.getElementById('mob-menu');
  if (ham && menu) {
    ham.addEventListener('click', () => {
      const isOpen = !menu.hasAttribute('hidden');
      if (isOpen) {
        menu.setAttribute('hidden', '');
        ham.setAttribute('aria-expanded', 'false');
        ham.setAttribute('aria-label', 'Open menu');
      } else {
        menu.removeAttribute('hidden');
        ham.setAttribute('aria-expanded', 'true');
        ham.setAttribute('aria-label', 'Close menu');
      }
    });

    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!ham.contains(e.target) && !menu.contains(e.target)) {
        menu.setAttribute('hidden', '');
        ham.setAttribute('aria-expanded', 'false');
        ham.setAttribute('aria-label', 'Open menu');
      }
    });

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !menu.hasAttribute('hidden')) {
        menu.setAttribute('hidden', '');
        ham.setAttribute('aria-expanded', 'false');
        ham.focus();
      }
    });
  }

  /* ── Stagger entrance ──────────────────────────────────────── */
  document.querySelectorAll('.stagger').forEach(el => el.classList.add('go'));

  /* ── Animate confidence bars ───────────────────────────────── */
  document.querySelectorAll('.conf-fill[data-target]').forEach(el => {
    setTimeout(() => { el.style.width = el.dataset.target + '%'; }, 150);
  });

  /* ── Animate token / bar-fill bars (staggered) ─────────────── */
  document.querySelectorAll('.bar-fill[data-target]').forEach((el, i) => {
    setTimeout(() => { el.style.width = el.dataset.target + '%'; }, 300 + i * 30);
  });

  /* ── Flash auto-dismiss ────────────────────────────────────── */
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 350ms ease, transform 350ms ease';
      el.style.opacity    = '0';
      el.style.transform  = 'translateY(-4px)';
      setTimeout(() => el.remove(), 370);
    }, 6000);
  });

  /* ── Details chevron rotation ──────────────────────────────── */
  document.querySelectorAll('details').forEach(d => {
    const chevron = d.querySelector('.fa-chevron-right');
    if (!chevron) return;
    d.addEventListener('toggle', () => {
      chevron.style.transform = d.open ? 'rotate(90deg)' : 'rotate(0deg)';
    });
  });

});
