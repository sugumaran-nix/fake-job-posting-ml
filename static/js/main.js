'use strict';

document.addEventListener('DOMContentLoaded', () => {

  // ── Nav hamburger ────────────────────────────────────────────
  const ham  = document.getElementById('hamburger');
  const menu = document.getElementById('mob-menu');
  if (ham && menu) {
    ham.addEventListener('click', () => menu.classList.toggle('hidden'));
    document.addEventListener('click', (e) => {
      if (!ham.contains(e.target) && !menu.contains(e.target))
        menu.classList.add('hidden');
    });
  }

  // ── Stagger entrance ─────────────────────────────────────────
  document.querySelectorAll('.stagger').forEach(el => el.classList.add('go'));

  // ── Animate confidence bars ──────────────────────────────────
  document.querySelectorAll('.conf-fill[data-target]').forEach(el => {
    setTimeout(() => { el.style.width = el.dataset.target + '%'; }, 150);
  });

  // ── Animate token bars — staggered ──────────────────────────
  document.querySelectorAll('.bar-fill[data-target]').forEach((el, i) => {
    setTimeout(() => { el.style.width = el.dataset.target + '%'; }, 300 + i * 30);
  });

  // ── Flash auto-dismiss ───────────────────────────────────────
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 350ms ease, transform 350ms ease';
      el.style.opacity    = '0';
      el.style.transform  = 'translateY(-4px)';
      setTimeout(() => el.remove(), 370);
    }, 5000);
  });

});
