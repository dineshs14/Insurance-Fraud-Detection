// main.js — Insurance Fraud Detection Framework (Enterprise Professional)

// ── Animate counter numbers ────────────────────────────────────────────────────
function animateCounter(el, target, duration = 1000, decimals = 0) {
  const start = performance.now();
  const startVal = 0;
  const formatter = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
  function update(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = startVal + (target - startVal) * eased;
    el.textContent = formatter.format(current);
    if (progress < 1) requestAnimationFrame(update);
    else el.textContent = formatter.format(target);
  }
  requestAnimationFrame(update);
}

document.querySelectorAll('[data-count]').forEach(el => {
  const target = parseFloat(el.dataset.count);
  const decimals = (el.dataset.decimals || 0) | 0;
  if (!isNaN(target)) animateCounter(el, target, 1000, decimals);
});

// ── Metric bar fill animation ─────────────────────────────────────────────────
document.querySelectorAll('.metric-fill').forEach(el => {
  const width = el.dataset.width || 0;
  setTimeout(() => { el.style.width = width + '%'; }, 150);
});

// ── Tier progress bar animation ───────────────────────────────────────────────
document.querySelectorAll('.tier-prog-fill, .tier-progress-fill').forEach(el => {
  const width = el.dataset.width || 0;
  setTimeout(() => { el.style.width = width + '%'; }, 250);
});

// ── Probability fill animation ────────────────────────────────────────────────
const probFill = document.getElementById('prob-fill');
if (probFill) {
  const pct = parseFloat(probFill.dataset.pct || 0);
  setTimeout(() => { probFill.style.width = Math.min(pct, 100) + '%'; }, 200);
}

// ── Upload drag & drop ─────────────────────────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const fnameDisplay = document.getElementById('fname-display');

if (uploadZone && fileInput) {
  uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      if (fnameDisplay) fnameDisplay.textContent = 'Selected: ' + e.dataTransfer.files[0].name;
    }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length && fnameDisplay) {
      fnameDisplay.textContent = 'Selected: ' + fileInput.files[0].name;
    }
  });
}

// ── Form submission loading state ─────────────────────────────────────────────
document.querySelectorAll('form').forEach(form => {
  form.addEventListener('submit', () => {
    const btn = form.querySelector('[type="submit"]');
    if (btn) {
      btn.disabled = true;
      btn.innerHTML = 'Processing analysis...';
    }
  });
});

// ── Active nav highlighting ────────────────────────────────────────────────────
const currentPath = window.location.pathname;
document.querySelectorAll('.nav-item').forEach(link => {
  if (link.getAttribute('href') === currentPath) link.classList.add('active');
  else if (link.getAttribute('href') !== '/' && currentPath.startsWith(link.getAttribute('href'))) link.classList.add('active');
});

// ── Chart.js global defaults ──────────────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#94A3B8';
  Chart.defaults.font.family = "'Inter', -apple-system, sans-serif";
  Chart.defaults.font.size = 11.5;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0F172A';
  Chart.defaults.plugins.tooltip.titleColor = '#F8FAFC';
  Chart.defaults.plugins.tooltip.bodyColor = '#CBD5E1';
  Chart.defaults.plugins.tooltip.borderColor = '#334155';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
}

// ── Render dashboard charts ────────────────────────────────────────────────────
function renderDashboardCharts(months, fraudTrend, legitTrend, typeDist, provDist, amtDist, tierCounts) {

  // 1. Trend Line Chart
  const trendCtx = document.getElementById('trendChart');
  if (trendCtx) {
    new Chart(trendCtx, {
      type: 'line',
      data: {
        labels: months,
        datasets: [
          {
            label: 'Legitimate Claims',
            data: legitTrend,
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59,130,246,0.06)',
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: '#3B82F6',
          },
          {
            label: 'Fraud Claims',
            data: fraudTrend,
            borderColor: '#EF4444',
            backgroundColor: 'rgba(239,68,68,0.06)',
            borderWidth: 2,
            fill: true,
            tension: 0.35,
            pointRadius: 3,
            pointBackgroundColor: '#EF4444',
          },
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top', align: 'end' },
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { maxTicksLimit: 8 } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
        }
      }
    });
  }

  // 2. Risk Tier Donut
  const tierCtx = document.getElementById('tierChart');
  if (tierCtx && tierCounts) {
    new Chart(tierCtx, {
      type: 'doughnut',
      data: {
        labels: Object.keys(tierCounts),
        datasets: [{
          data: Object.values(tierCounts),
          backgroundColor: ['#22C55E', '#F59E0B', '#EF4444'],
          borderColor: '#111827',
          borderWidth: 2,
          hoverOffset: 4,
        }]
      },
      options: {
        responsive: true,
        cutout: '72%',
        plugins: {
          legend: { position: 'bottom', labels: { padding: 12 } }
        }
      }
    });
  }

  // 3. Claim Type Horizontal Bar
  const typeCtx = document.getElementById('typeChart');
  if (typeCtx && typeDist) {
    new Chart(typeCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(typeDist),
        datasets: [{
          data: Object.values(typeDist),
          backgroundColor: '#3B82F6',
          borderRadius: 4,
          borderSkipped: false,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // 4. Provider Fraud Rate
  const provCtx = document.getElementById('provChart');
  if (provCtx && provDist) {
    new Chart(provCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(provDist),
        datasets: [{
          label: 'Fraud Rate %',
          data: Object.values(provDist),
          backgroundColor: 'rgba(245,158,11,0.85)',
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: ctx => ` ${ctx.parsed.y}% fraud rate` }
          }
        },
        scales: {
          x: { grid: { display: false } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' },
               ticks: { callback: v => v + '%' } }
        }
      }
    });
  }

  // 5. Amount Distribution
  const amtCtx = document.getElementById('amtChart');
  if (amtCtx && amtDist) {
    new Chart(amtCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(amtDist),
        datasets: [{
          label: 'Fraud Claims',
          data: Object.values(amtDist),
          backgroundColor: 'rgba(139,92,246,0.85)',
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true }
        }
      }
    });
  }
}
