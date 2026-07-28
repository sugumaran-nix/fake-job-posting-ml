/* ═══════════════════════════════════════════════════════════
   JobGuard — main.js
   ═══════════════════════════════════════════════════════════ */

/* ── Navbar scroll state ─────────────────────────────────── */
(function () {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  const toggle = () => nav.classList.toggle('scrolled', window.scrollY > 20);
  toggle();
  window.addEventListener('scroll', toggle, { passive: true });
})();

/* ── Counter animation (home stats) ──────────────────────── */
(function () {
  const els = document.querySelectorAll('[data-target]');
  if (!els.length) return;

  const animate = (el) => {
    const target = parseInt(el.dataset.target, 10);
    if (isNaN(target)) return;
    const duration = 1200;
    const step = 16;
    const steps = duration / step;
    const inc = target / steps;
    let current = 0;
    const timer = setInterval(() => {
      current = Math.min(current + inc, target);
      el.textContent = Math.floor(current).toLocaleString();
      if (current >= target) clearInterval(timer);
    }, step);
  };

  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { animate(e.target); obs.unobserve(e.target); } });
    }, { threshold: 0.4 });
    els.forEach(el => obs.observe(el));
  } else {
    els.forEach(animate);
  }
})();

/* ── Classify form: description char counter ─────────────── */
(function () {
  const desc = document.getElementById('desc');
  const cc   = document.getElementById('cc');
  if (!desc || !cc) return;

  const update = () => {
    const len  = desc.value.length;
    const good = len >= 150;
    cc.innerHTML = `
      <div class="char-rail ${good ? 'good' : len > 50 ? 'warn' : ''}">
        <div class="char-fill" style="width:${Math.min(len / 5, 100)}%"></div>
      </div>
      <span class="char-count-label">
        ${len} chars ${good ? '— <span style="color:var(--legit)">good length</span>' : '— aim for 150+'}
      </span>`;
  };

  window.updateCC = update;
  desc.addEventListener('input', update);
  update();
})();

/* ── Classify form: submit spinner ───────────────────────── */
(function () {
  const form = document.getElementById('cform');
  const btn  = document.getElementById('sbtn');
  if (!form || !btn) return;
  form.addEventListener('submit', () => {
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analysing…';
  });
})();

/* ── Result: animate confidence fill bar ─────────────────── */
(function () {
  document.querySelectorAll('.conf-fill[data-target]').forEach(el => {
    const target = parseFloat(el.dataset.target) || 0;
    el.style.width = '0%';
    setTimeout(() => { el.style.width = target + '%'; }, 250);
  });
})();

/* ── Result: animate influence bars ─────────────────────── */
(function () {
  const bars = document.querySelectorAll('.inf-bar-fraud, .inf-bar-legit');
  if (!bars.length) return;
  bars.forEach(b => {
    const w = b.style.width;
    b.style.width = '0';
    setTimeout(() => { b.style.width = w; }, 500);
  });
})();

/* ── Auto-dismiss flash messages ─────────────────────────── */
(function () {
  document.querySelectorAll('.flash').forEach(f => {
    setTimeout(() => {
      f.style.transition = 'opacity .5s';
      f.style.opacity = '0';
      setTimeout(() => f.remove(), 500);
    }, 5000);
  });
})();
