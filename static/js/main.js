/* Fake Job Posting Prediction — Enhanced main.js */

/* ── Counter animation ─────────────────────────────────── */
function animateCounters() {
  document.querySelectorAll('.stat-n[data-target]').forEach(el => {
    const target = parseInt(el.getAttribute('data-target')) || 0;
    if (target === 0) { el.textContent = '0'; return; }
    const duration = 1600, steps = 70;
    let count = 0;
    // Ease-out curve
    const ease = t => 1 - Math.pow(1 - t, 3);
    const interval = setInterval(() => {
      count++;
      const progress = ease(count / steps);
      el.textContent = Math.floor(progress * target).toLocaleString();
      if (count >= steps) {
        el.textContent = target.toLocaleString();
        clearInterval(interval);
      }
    }, duration / steps);
  });
}

/* ── Confidence bar animation ──────────────────────────── */
function animateConfBar() {
  const bar = document.querySelector('.conf-fill');
  if (!bar) return;
  const targetW = bar.style.width;
  bar.style.width = '0%';
  bar.style.transition = 'none';
  /* Small delay so scroll-reveal opacity animation finishes first */
  setTimeout(() => {
    bar.style.transition = 'width 1.6s cubic-bezier(.4,0,.2,1)';
    bar.style.width = targetW;
  }, 300);
}

/* ── Scroll reveal ─────────────────────────────────────── */
function initReveal() {
  const selector = '.step-card,.flag-card,.stat-box,.acard,.metric-card,.adv-card,.expl-section,.fcard,.about-banner,.pipeline-box';
  const els = document.querySelectorAll(selector);
  if (!els.length) return;

  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (e.isIntersecting) {
        // Stagger siblings
        const siblings = [...e.target.parentElement.querySelectorAll(selector)];
        const idx = siblings.indexOf(e.target);
        setTimeout(() => {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
        }, idx * 65);
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.07 });

  els.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(24px)';
    el.style.transition = 'opacity .5s ease, transform .5s ease';
    obs.observe(el);
  });
}

/* ── Navbar scroll effect ──────────────────────────────── */
function initNavbar() {
  const nav = document.querySelector('.navbar');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    const y = window.scrollY;
    if (y > 20) {
      nav.style.background = '#ffffff';
      nav.style.boxShadow = '0 1px 0 rgba(15,23,42,0.06), 0 4px 20px rgba(15,23,42,0.10)';
    } else {
      nav.style.background = '#ffffff';
      nav.style.boxShadow = '0 1px 0 rgba(15,23,42,0.04), 0 2px 12px rgba(15,23,42,0.07)';
    }
  }, { passive: true });
}

/* ── Influence bars animate on view ───────────────────── */
function initInfluenceBars() {
  const bars = document.querySelectorAll('.inf-bar');
  if (!bars.length) return;

  /* Save original widths FIRST, then zero */
  bars.forEach(bar => {
    bar.setAttribute('data-w', bar.style.width || '0%');
    bar.style.width = '0%';
    bar.style.transition = 'none';
  });

  const section = document.querySelector('.influence-grid');
  if (!section) return;

  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        bars.forEach((bar, i) => {
          setTimeout(() => {
            bar.style.transition = 'width .9s cubic-bezier(.4,0,.2,1)';
            bar.style.width = bar.getAttribute('data-w');
          }, i * 50);
        });
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });
  obs.observe(section);
}

/* ── Mini rail animate ─────────────────────────────────── */
function initMiniRails() {
  document.querySelectorAll('.mini-fill').forEach(el => {
    const w = el.style.width;
    if (!w || w === '0%' || w === '0') return;
    el.style.width = '0';
    el.style.transition = 'none';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        el.style.transition = 'width .8s ease';
        el.style.width = w;
      });
    });
  });
}

/* ── Flash auto-dismiss ────────────────────────────────── */
function initFlash() {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => {
      el.style.transition = 'opacity .5s ease, transform .5s ease';
      el.style.opacity = '0';
      el.style.transform = 'translateY(-8px)';
      setTimeout(() => el.remove(), 500);
    }, 4000);
  });
}

/* ── Character counter progress bar ─────────────────────── */
function initCharBar() {
  const desc = document.getElementById('desc');
  const ccSpan = document.getElementById('cc');
  const row = ccSpan ? ccSpan.closest('.char-row') : null;
  if (!desc || !ccSpan || !row) return;

  const MAX = 2000;

  /* Rebuild char-row into bar layout */
  row.innerHTML = `
    <div class="char-rail" id="charRail">
      <div class="char-fill" id="charFill"></div>
    </div>
    <span class="char-count-label"><span id="cc">0</span> / ${MAX} chars</span>
  `;

  const fill  = document.getElementById('charFill');
  const rail  = document.getElementById('charRail');
  const newCc = document.getElementById('cc');

  function update() {
    const len = desc.value.length;
    newCc.textContent = len.toLocaleString();
    const pct = Math.min((len / MAX) * 100, 100);
    fill.style.width = pct + '%';
    rail.className = 'char-rail';
    if (pct >= 80) rail.classList.add('warn');
    else if (pct >= 30) rail.classList.add('good');
  }

  desc.addEventListener('input', update);
  update();

  /* Override global updateCC used by classify.html inline script */
  window.updateCC = update;
}

/* ── Submit button loading state ──────────────────────── */
function initSubmitBtn() {
  const form = document.getElementById('cform');
  if (!form) return;
  form.addEventListener('submit', () => {
    const b = document.getElementById('sbtn');
    if (!b) return;
    b.innerHTML = '<span style="display:inline-block;animation:spin .8s linear infinite">⟳</span> Analysing…';
    b.disabled = true;
    b.style.opacity = '.8';
  });
}

/* Spinner keyframe */
const style = document.createElement('style');
style.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
document.head.appendChild(style);

/* ── Init ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  animateCounters();
  animateConfBar();
  initReveal();
  initNavbar();
  initInfluenceBars();
  initMiniRails();
  initFlash();
  initCharBar();
  initSubmitBtn();
});
