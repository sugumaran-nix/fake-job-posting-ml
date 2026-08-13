/* JobGuard — main.js */
'use strict';

// ── Nav hamburger ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const ham    = document.getElementById('hamburger');
  const menu   = document.getElementById('mob-menu');
  if (ham && menu) {
    ham.addEventListener('click', () => menu.classList.toggle('hidden'));
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!ham.contains(e.target) && !menu.contains(e.target)) {
        menu.classList.add('hidden');
      }
    });
  }

  // ── Stagger parent trigger ───────────────────────────────────
  document.querySelectorAll('.stagger-parent').forEach(el =>
    el.classList.add('animate')
  );

  // ── Animate conf-fill on page load ──────────────────────────
  document.querySelectorAll('.conf-fill[data-target]').forEach(el => {
    setTimeout(() => { el.style.width = el.dataset.target + '%'; }, 180);
  });

  // ── Stagger bar fills ────────────────────────────────────────
  document.querySelectorAll('.bar-fill').forEach((el, i) => {
    const target = el.style.width;
    el.style.width = '0%';
    setTimeout(() => { el.style.width = target; }, 350 + i * 25);
  });

  // ── Flash auto-dismiss ───────────────────────────────────────
  document.querySelectorAll('.flash-msg').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity 400ms ease, transform 400ms ease';
      el.style.opacity    = '0';
      el.style.transform  = 'translateY(-6px)';
      setTimeout(() => el.remove(), 420);
    }, 4500);
  });
});
