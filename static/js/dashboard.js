/* ── dashboard.js — shared across all pages ── */

const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor:  'transparent',
  font:  { color: '#e8eaf0', family: 'Segoe UI, system-ui, sans-serif', size: 12 },
  margin: { t: 36, r: 16, b: 48, l: 52 },
  xaxis: { gridcolor: '#2d3148', zerolinecolor: '#2d3148', color: '#8b92a9' },
  yaxis: { gridcolor: '#2d3148', zerolinecolor: '#2d3148', color: '#8b92a9' },
  legend: { bgcolor: 'transparent', font: { color: '#8b92a9' } },
};

// ─────────────────────────────────────────────────────────────────────────────
// Navbar hydration (runs on every page)
// ─────────────────────────────────────────────────────────────────────────────
async function hydrateNavbar() {
  try {
    const [dsRes, alertRes] = await Promise.all([
      fetch('/api/active-dataset'),
      fetch('/api/alerts'),
    ]);
    const ds    = await dsRes.json();
    const alerts = await alertRes.json();

    const badge = document.getElementById('nav-dataset-badge');
    if (badge && ds.filename) badge.textContent = 'Active: ' + ds.filename;

    const cnt = document.getElementById('nav-alert-count');
    if (cnt) {
      const unread = alerts.unread || 0;
      cnt.textContent = unread;
      cnt.style.display = unread > 0 ? 'inline' : 'none';
    }
  } catch (_) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast
// ─────────────────────────────────────────────────────────────────────────────
function showToast(message, type = 'success') {
  const colors = { success: '#2ecc71', error: '#e74c3c', warning: '#f39c12', info: '#4f8ef7' };
  const id = 'toast-' + Date.now();
  const html = `
    <div id="${id}" class="toast align-items-center border-0 show" role="alert" style="border-left:4px solid ${colors[type]} !important">
      <div class="d-flex">
        <div class="toast-body">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    </div>`;
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    document.body.appendChild(container);
  }
  container.insertAdjacentHTML('beforeend', html);
  setTimeout(() => document.getElementById(id)?.remove(), 4000);
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading overlay
// ─────────────────────────────────────────────────────────────────────────────
function showLoading(text = 'Processing…') {
  let el = document.getElementById('loading-overlay');
  if (!el) {
    el = document.createElement('div');
    el.id = 'loading-overlay';
    el.innerHTML = `<div class="spinner-border"></div><div class="load-text">${text}</div>`;
    document.body.appendChild(el);
  }
  el.querySelector('.load-text').textContent = text;
  el.classList.add('show');
}
function hideLoading() {
  document.getElementById('loading-overlay')?.classList.remove('show');
}

// ─────────────────────────────────────────────────────────────────────────────
// KPI card count-up animation
// ─────────────────────────────────────────────────────────────────────────────
function countUp(el, target, decimals = 0, suffix = '') {
  const start = 0, duration = 700, step = 16;
  const steps = duration / step;
  let current = start, i = 0;
  const inc = (target - start) / steps;
  const timer = setInterval(() => {
    current += inc; i++;
    el.textContent = current.toFixed(decimals) + suffix;
    if (i >= steps) { el.textContent = target.toFixed(decimals) + suffix; clearInterval(timer); }
  }, step);
}

// ─────────────────────────────────────────────────────────────────────────────
// updateKPICards
// ─────────────────────────────────────────────────────────────────────────────
function updateKPICards(perfData, driftData, alertData) {
  // Card 1 — best model
  const c1val = document.getElementById('kpi-model-name');
  const c1sub = document.getElementById('kpi-model-type');
  if (c1val && perfData) {
    c1val.textContent = perfData.best_model || '—';
    const ensembleKw = ['Voting','Stacking','Bagging','AdaBoost'];
    const isEnsemble = ensembleKw.some(k => (perfData.best_model||'').includes(k));
    if (c1sub) c1sub.textContent = isEnsemble ? 'Ensemble' : 'Individual';
  }

  // Card 2 — accuracy
  const c2val = document.getElementById('kpi-accuracy');
  const c2sub = document.getElementById('kpi-accuracy-sub');
  if (c2val && perfData?.model_results?.length) {
    const rows = perfData.model_results;
    const best = rows[0];
    const scoreKey = ['f1','accuracy','r2'].find(k => best[k] !== undefined && best[k] !== 'N/A');
    const score = parseFloat(best[scoreKey] || 0);
    countUp(c2val, score * 100, 1, '%');
    if (c2sub && rows[1]) {
      const s2 = parseFloat(rows[1][scoreKey] || 0);
      c2sub.textContent = `2nd best: ${(s2*100).toFixed(1)}%`;
    }
  }

  // Card 3 — drift
  const c3val = document.getElementById('kpi-drift-status');
  const c3card = document.getElementById('kpi-drift-card');
  if (c3val && driftData) {
    const sev = driftData.psi_severity || 'No Drift';
    c3val.textContent = sev;
    if (c3card) {
      c3card.className = c3card.className.replace(/\b(green|orange|red)\b/g,'');
      c3card.classList.add(sev === 'No Drift' ? 'green' : sev === 'Moderate' ? 'orange' : 'red');
    }
  }

  // Card 4 — alerts
  const c4val = document.getElementById('kpi-alert-count');
  const c4sub = document.getElementById('kpi-alert-sub');
  if (c4val && alertData) {
    const all = alertData.alerts || [];
    countUp(c4val, all.length, 0);
    const crit = all.filter(a => a.level === 'CRITICAL').length;
    const warn = all.filter(a => a.level === 'WARNING').length;
    if (c4sub) c4sub.textContent = `${crit} Critical, ${warn} Warning`;
    const c4card = document.getElementById('kpi-alert-card');
    if (c4card) {
      c4card.className = c4card.className.replace(/\b(green|yellow|red)\b/g,'');
      c4card.classList.add(crit > 0 ? 'red' : warn > 0 ? 'yellow' : 'green');
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// renderPerformanceChart
// ─────────────────────────────────────────────────────────────────────────────
function renderPerformanceChart(perfData) {
  const el = document.getElementById('perf-chart');
  if (!el || !perfData?.model_results) return;

  const rows = perfData.model_results.filter(r => r.f1 !== 'N/A' && r.f1 !== undefined);
  const scoreKey = perfData.problem_type === 'regression' ? 'r2' : 'f1';

  // Simulate batch degradation for the line chart
  const batches = Array.from({length: 8}, (_, i) => i + 1);
  const bestScore = parseFloat(rows[0]?.[scoreKey] || 0.9);
  const scores = batches.map((b, i) => +(bestScore - i * 0.012 + (Math.random() - 0.5) * 0.01).toFixed(4));

  const traces = [{
    x: batches, y: scores,
    mode: 'lines+markers',
    name: perfData.best_model || 'Best Model',
    line: { color: '#4f8ef7', width: 2.5 },
    marker: { size: 6, color: '#4f8ef7' },
  }];

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: 'Model Performance Over Incoming Data Batches', font: { size: 13, color: '#e8eaf0' } },
    xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'Batch Number', dtick: 1 },
    yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: scoreKey.toUpperCase() + ' Score', range: [0.5, 1.0] },
    shapes: [
      { type:'line', x0:1, x1:8, y0:0.75, y1:0.75, line:{ color:'#e74c3c', dash:'dash', width:1.5 } },
      { type:'line', x0:1, x1:8, y0:0.85, y1:0.85, line:{ color:'#f39c12', dash:'dash', width:1.5 } },
    ],
    annotations: [
      { x:8, y:0.75, text:'Critical', showarrow:false, font:{color:'#e74c3c', size:10}, xanchor:'right' },
      { x:8, y:0.85, text:'Warning',  showarrow:false, font:{color:'#f39c12', size:10}, xanchor:'right' },
    ],
  };

  Plotly.react(el, traces, layout, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────────────────────
// renderDriftChart  (feature PSI horizontal bars)
// ─────────────────────────────────────────────────────────────────────────────
function renderDriftChart(driftData) {
  const el = document.getElementById('drift-bar-chart');
  if (!el || !driftData?.feature_psi?.length) return;

  const features = driftData.feature_psi;
  const names  = features.map(f => f.feature);
  const values = features.map(f => f.psi);
  const colors = values.map(v => v > 0.2 ? '#e74c3c' : v > 0.1 ? '#f39c12' : '#2ecc71');

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    title: { text: 'Per-Feature Drift Scores (PSI)', font: { size: 13, color: '#e8eaf0' } },
    xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: 'PSI Score' },
    yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, automargin: true },
    margin: { ...PLOTLY_LAYOUT_BASE.margin, l: 120 },
    height: Math.max(280, names.length * 28 + 80),
  };

  Plotly.react(el, [{
    type: 'bar', orientation: 'h',
    x: values, y: names,
    marker: { color: colors },
    text: values.map(v => v.toFixed(4)),
    textposition: 'outside',
    textfont: { color: '#8b92a9', size: 10 },
  }], layout, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────────────────────
// fetchDashboardData  (main refresh loop)
// ─────────────────────────────────────────────────────────────────────────────
async function fetchDashboardData() {
  try {
    const [perfRes, driftRes, alertRes] = await Promise.all([
      fetch('/api/performance-metrics'),
      fetch('/api/drift-status'),
      fetch('/api/alerts'),
    ]);
    const perf  = perfRes.ok  ? await perfRes.json()  : null;
    const drift = driftRes.ok ? await driftRes.json() : null;
    const alert = alertRes.ok ? await alertRes.json() : null;

    if (perf)  renderPerformanceChart(perf);
    if (drift) renderDriftChart(drift);
    updateKPICards(perf, drift, alert);
    if (typeof renderAlertsTable === 'function' && alert) renderAlertsTable(alert.alerts || []);
    if (typeof renderDriftSummary === 'function' && drift) renderDriftSummary(drift);
    hydrateNavbar();
  } catch (e) { console.warn('fetchDashboardData error', e); }
}

// ─────────────────────────────────────────────────────────────────────────────
// simulateNextBatch
// ─────────────────────────────────────────────────────────────────────────────
async function simulateNextBatch() {
  showLoading('Simulating next batch…');
  try {
    const res  = await fetch('/api/simulate-batch', { method: 'POST' });
    const data = await res.json();
    hideLoading();
    if (!res.ok) { showToast(data.error || 'Simulation failed', 'error'); return; }
    showToast(`Batch ${data.batch_id} processed — PSI: ${data.overall_psi.toFixed(4)}`, data.alert ? 'warning' : 'success');
    fetchDashboardData();
  } catch (e) { hideLoading(); showToast('Simulation error', 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// retrainModel
// ─────────────────────────────────────────────────────────────────────────────
async function retrainModel() {
  showLoading('Retraining model…');
  try {
    const res  = await fetch('/api/retrain', { method: 'POST' });
    const data = await res.json();
    hideLoading();
    if (!res.ok) { showToast(data.error || 'Retrain failed', 'error'); return; }
    showToast(`Model retrained! New best: ${data.best_model} (score: ${(data.score*100).toFixed(1)}%)`, 'success');
    fetchDashboardData();
  } catch (e) { hideLoading(); showToast('Retrain error', 'error'); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-refresh every 30 s
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  hydrateNavbar();
  if (typeof fetchDashboardData === 'function') {
    fetchDashboardData();
    setInterval(fetchDashboardData, 30000);
  }
});
