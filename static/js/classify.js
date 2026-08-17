const SAMPLES = {
  fraud: {
    job_title: 'Work From Home – Easy Data Entry Operator',
    company: 'QuickEarn Digital Pvt Ltd',
    location: 'Remote, Anywhere in India',
    salary: '₹50,000–₹1,20,000/month guaranteed',
    website: 'http://quickearnnow.in.free.hosting.com',
    description: 'URGENT HIRING for a Work From Home Data Entry Operator job. Role: online typing and simple data entry work. Responsibilities: copy information into forms, submit daily reports, and complete assigned tasks. Requirements: no degree needed, basic smartphone skills, and a bank account. Earn guaranteed income of ₹50,000 per month. Pay a registration fee of ₹799 to receive the starter kit. Apply now before the offer expires!',
    requirements: 'No degree needed. Registration fee of ₹799 compulsory to receive the joining kit.'
  },
  legit: {
    job_title: 'Junior Backend Developer (Python)',
    company: 'Zoho Corporation',
    location: 'Chennai, Tamil Nadu',
    salary: '₹4–7 LPA',
    website: 'https://careers.zohocorp.com',
    description: 'We are looking for a motivated Junior Backend Developer to join our engineering team in Chennai. You will work on REST APIs, microservices, and internal tooling. Responsibilities include writing clean, testable Python code, participating in code reviews, contributing to Agile sprints, and maintaining technical documentation. We offer a competitive salary, health insurance, PF, gratuity, and structured L&D support. Five days a week, work from office. Apply by sending your resume to careers@zohocorp.com.',
    requirements: 'B.E./B.Tech/MCA in Computer Science or related. Proficiency in Python 3.x, Flask or FastAPI, Git, SQL. 0–2 years experience or strong internship background.'
  }
};

function updateCounter() {
  const description = document.getElementById('f_description');
  const fill = document.getElementById('char-fill');
  const counter = document.getElementById('desc-counter');
  if (!description) return;
  const length = description.value.length;
  const percentage = Math.min((length / 500) * 100, 100);
  if (fill) {
    fill.style.width = `${percentage}%`;
    fill.style.background = length >= 150 ? 'var(--legit)' : length > 50 ? 'var(--warn)' : 'var(--t4)';
  }
  if (counter) {
    if (length === 0) {
      counter.textContent = '0 characters';
      counter.style.color = '';
    } else if (length < 150) {
      counter.textContent = `${length} characters — ${150 - length} more recommended`;
      counter.style.color = 'var(--warn-text)';
    } else {
      counter.textContent = `${length} characters — ready to review`;
      counter.style.color = 'var(--legit-text)';
    }
  }
}

function clearFormFields() {
  const form = document.getElementById('cform');
  if (!form) return;
  form.querySelectorAll('input:not([type="hidden"]), textarea, select').forEach((field) => {
    field.value = '';
  });
  resetFormState();
}

function resetFormState() {
  const description = document.getElementById('f_description');
  if (description) {
    description.style.borderColor = '';
    description.style.boxShadow = '';
  }
  const button = document.getElementById('sbtn');
  if (button) {
    button.disabled = false;
    button.innerHTML = '<i class="fa-solid fa-magnifying-glass" aria-hidden="true" style="font-size:13px;"></i> Get my risk report';
  }
  const overlay = document.getElementById('loading-overlay');
  if (overlay) overlay.style.display = 'none';
  window.requestAnimationFrame(updateCounter);
}

function loadSample(type) {
  const sample = SAMPLES[type];
  if (!sample) return;
  ['job_title', 'company', 'location', 'salary', 'website', 'requirements'].forEach((field) => {
    const element = document.getElementById(`f_${field}`);
    if (element) element.value = sample[field] || '';
  });
  const description = document.getElementById('f_description');
  if (description) {
    description.value = sample.description;
    description.dispatchEvent(new Event('input', { bubbles: true }));
    window.setTimeout(() => description.scrollIntoView({ behavior: 'smooth', block: 'center' }), 80);
  }
}

let classifyInitialized = false;

function initClassify() {
  if (classifyInitialized) return;
  const form = document.getElementById('cform');
  if (!form) return;
  classifyInitialized = true;

  // Delegate clicks so the handlers remain reliable after navigation and bfcache restores.
  document.addEventListener('click', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const sampleButton = target?.closest('[data-sample]');
    if (sampleButton) {
      event.preventDefault();
      loadSample(sampleButton.dataset.sample);
      return;
    }
    const clearButton = target?.closest('#cbtn');
    if (clearButton) {
      event.preventDefault();
      clearFormFields();
    }
  });

  form.addEventListener('reset', (event) => {
    event.preventDefault();
    clearFormFields();
  });
  document.getElementById('f_description')?.addEventListener('input', updateCounter);
  form.addEventListener('submit', (event) => {
    const description = document.getElementById('f_description');
    if (!description || description.value.trim().length < 20) {
      event.preventDefault();
      description?.focus();
      if (description) {
        description.style.borderColor = 'var(--fraud)';
        description.style.boxShadow = '0 0 0 3px var(--fraud-dim)';
      }
      const counter = document.getElementById('desc-counter');
      if (counter) {
        counter.textContent = 'Please paste a job description before analysing.';
        counter.style.color = 'var(--fraud-text)';
      }
      return;
    }
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.style.display = 'flex';
    const button = document.getElementById('sbtn');
    if (button) {
      button.disabled = true;
      button.innerHTML = '<i class="fa-solid fa-spinner fa-spin" aria-hidden="true" style="font-size:13px;"></i> Analysing…';
    }
  });
  updateCounter();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initClassify, { once: true });
} else {
  initClassify();
}

window.addEventListener('pageshow', (event) => {
  if (event.persisted) resetFormState();
});
