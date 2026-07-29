/* JobGuard — main.js */

/* ── Navbar scroll shadow ──────────────────────── */
(function () {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  const fn = () => {
    if (window.scrollY > 24) {
      nav.style.boxShadow = '0 1px 0 rgba(59,130,246,.15), 0 4px 24px rgba(0,0,0,.5)';
    } else {
      nav.style.boxShadow = '';
    }
  };
  fn();
  window.addEventListener('scroll', fn, { passive: true });
})();

/* ── Hamburger ─────────────────────────────────── */
(function () {
  const btn  = document.getElementById('hamburger');
  const menu = document.getElementById('mob-menu');
  if (!btn || !menu) return;
  btn.addEventListener('click', () => {
    menu.classList.toggle('hidden');
  });
})();

/* ── Counter animation (home stats) ───────────── */
(function () {
  const els = document.querySelectorAll('[data-target]');
  if (!els.length) return;
  const animate = el => {
    const target = parseInt(el.dataset.target, 10);
    if (isNaN(target)) return;
    const duration = 1200, step = 16;
    const inc = target / (duration / step);
    let cur = 0;
    const t = setInterval(() => {
      cur = Math.min(cur + inc, target);
      el.textContent = Math.floor(cur).toLocaleString();
      if (cur >= target) clearInterval(t);
    }, step);
  };
  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) { animate(e.target); obs.unobserve(e.target); } });
    }, { threshold: 0.4 });
    els.forEach(el => obs.observe(el));
  } else {
    els.forEach(animate);
  }
})();

/* ── Confidence bar (result page) ─────────────── */
(function () {
  document.querySelectorAll('.conf-fill[data-target]').forEach(el => {
    el.style.width = '0%';
    setTimeout(() => { el.style.width = el.dataset.target + '%'; }, 250);
  });
})();

/* ── Flash auto-dismiss ────────────────────────── */
(function () {
  document.querySelectorAll('.flash-msg').forEach(f => {
    setTimeout(() => {
      f.style.transition = 'opacity .5s';
      f.style.opacity = '0';
      setTimeout(() => f.remove(), 500);
    }, 5000);
  });
})();
