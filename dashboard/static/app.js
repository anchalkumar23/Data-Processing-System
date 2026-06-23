/**
 * dashboard/static/app.js
 * ────────────────────────
 * Real-time monitoring dashboard logic.
 *
 * Architecture:
 *   - Polls /api/v1/metrics every second for aggregate stats (KPIs)
 *   - Polls /api/v1/predictions every second for the timeline chart
 *   - Renders three Chart.js charts: defect rate line, latency histogram, class donut
 *   - Updates the predictions table with the latest 20 rows
 *   - Handles image upload → /api/v1/predict and displays the result
 *
 * We use plain fetch() with async/await rather than a framework to keep
 * the code readable and dependency-free.
 */

"use strict";

// ── Configuration ─────────────────────────────────────────────────────────────

const API_BASE       = "";           // same origin; change if API runs elsewhere
const POLL_INTERVAL  = 1000;         // milliseconds between metric polls
const HISTORY_LEN    = 60;           // points to keep in the rolling line chart

// ── State ─────────────────────────────────────────────────────────────────────

let isPaused         = false;
let pollTimerId      = null;
let defectHistory    = [];           // rolling array of defect rate values
let latencyHistory   = [];           // rolling array of latency values (ms)
let prevMetrics      = null;         // used to calculate KPI trend arrows

// ── Chart instances ───────────────────────────────────────────────────────────

let chartDefect  = null;
let chartLatency = null;
let chartDonut   = null;

// ── DOM references ────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);

const els = {
  statusDot:        $("statusDot"),
  statusLabel:      $("statusLabel"),
  valTotal:         $("valTotal"),
  valDefect:        $("valDefect"),
  valLatency:       $("valLatency"),
  valThroughput:    $("valThroughput"),
  trendTotal:       $("trendTotal"),
  trendDefect:      $("trendDefect"),
  trendLatency:     $("trendLatency"),
  trendThroughput:  $("trendThroughput"),
  badgeDefectRate:  $("badgeDefectRate"),
  badgeP95:         $("badgeP95"),
  donutCentreVal:   $("donutCentreVal"),
  predTableBody:    $("predTableBody"),
  btnTogglePause:   $("btnTogglePause"),
  iconPause:        $("iconPause"),
  iconPlay:         $("iconPlay"),
  btnUploadDemo:    $("btnUploadDemo"),
  uploadPanel:      $("uploadPanel"),
  btnCloseUpload:   $("btnCloseUpload"),
  uploadArea:       $("uploadArea"),
  fileInput:        $("fileInput"),
  uploadResult:     $("uploadResult"),
};


// ── Chart setup ───────────────────────────────────────────────────────────────

/**
 * Creates the defect rate timeline chart (line chart).
 * Fills the chart body of #chartDefectRate.
 */
function initDefectRateChart() {
  const ctx = $("chartDefectRate").getContext("2d");

  const gradient = ctx.createLinearGradient(0, 0, 0, 200);
  gradient.addColorStop(0, "rgba(245, 158, 11, 0.3)");
  gradient.addColorStop(1, "rgba(245, 158, 11, 0)");

  chartDefect = new Chart(ctx, {
    type: "line",
    data: {
      labels:   [],
      datasets: [{
        label:           "Defect Rate",
        data:            [],
        borderColor:     "#f59e0b",
        backgroundColor: gradient,
        borderWidth:     2,
        pointRadius:     0,
        pointHoverRadius: 4,
        tension:         0.4,
        fill:            true,
      }]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 300 },
      interaction:         { intersect: false, mode: "index" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(17, 24, 39, 0.95)",
          borderColor:     "rgba(255,255,255,0.07)",
          borderWidth:     1,
          titleColor:      "#f0f4ff",
          bodyColor:       "#8b9ab2",
          callbacks: {
            label: ctx => `Defect rate: ${(ctx.parsed.y * 100).toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          grid:    { color: "rgba(255,255,255,0.03)" },
          ticks:   { color: "#4a5568", font: { size: 10 }, maxTicksLimit: 8 },
        },
        y: {
          min:  0,
          max:  1,
          grid: { color: "rgba(255,255,255,0.03)" },
          ticks: {
            color: "#4a5568",
            font:  { size: 10 },
            callback: v => `${(v * 100).toFixed(0)}%`,
          },
        },
      },
    },
  });
}

/**
 * Creates the latency histogram (bar chart).
 */
function initLatencyHistChart() {
  const ctx = $("chartLatencyHist").getContext("2d");

  chartLatency = new Chart(ctx, {
    type: "bar",
    data: {
      labels:   ["<5", "5-10", "10-20", "20-30", "30-50", "50-100", ">100"],
      datasets: [{
        label:           "Count",
        data:            [0, 0, 0, 0, 0, 0, 0],
        backgroundColor: [
          "rgba(16, 185, 129, 0.7)",
          "rgba(16, 185, 129, 0.7)",
          "rgba(59, 130, 246, 0.7)",
          "rgba(59, 130, 246, 0.7)",
          "rgba(245, 158, 11, 0.7)",
          "rgba(239, 68, 68, 0.7)",
          "rgba(239, 68, 68, 0.7)",
        ],
        borderRadius:    4,
        borderSkipped:   false,
      }]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 400 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(17, 24, 39, 0.95)",
          borderColor:     "rgba(255,255,255,0.07)",
          borderWidth:     1,
          titleColor:      "#f0f4ff",
          bodyColor:       "#8b9ab2",
          callbacks: {
            title: items => `Latency: ${items[0].label}ms`,
            label: ctx   => `${ctx.parsed.y} requests`,
          },
        },
      },
      scales: {
        x: {
          grid:  { display: false },
          ticks: { color: "#4a5568", font: { size: 10 } },
        },
        y: {
          grid:  { color: "rgba(255,255,255,0.03)" },
          ticks: { color: "#4a5568", font: { size: 10 }, stepSize: 1 },
        },
      },
    },
  });
}

/**
 * Creates the class distribution donut chart.
 */
function initDonutChart() {
  const ctx = $("chartDonut").getContext("2d");

  chartDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels:   ["Normal", "Defective"],
      datasets: [{
        data:            [50, 50],
        backgroundColor: ["rgba(16, 185, 129, 0.8)", "rgba(239, 68, 68, 0.8)"],
        borderColor:     ["rgba(16, 185, 129, 0.3)", "rgba(239, 68, 68, 0.3)"],
        borderWidth:     1,
        hoverBackgroundColor: ["#10b981", "#ef4444"],
      }]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: true,
      cutout:              "72%",
      animation:           { duration: 600 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(17, 24, 39, 0.95)",
          borderColor:     "rgba(255,255,255,0.07)",
          borderWidth:     1,
          titleColor:      "#f0f4ff",
          bodyColor:       "#8b9ab2",
        },
      },
    },
  });
}


// ── Data fetching ─────────────────────────────────────────────────────────────

/**
 * Fetches aggregate metrics from /api/v1/metrics and updates the UI.
 * Returns the parsed JSON on success, or null on error.
 */
async function fetchMetrics() {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/metrics`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    setOffline();
    return null;
  }
}

/**
 * Fetches recent predictions from /api/v1/predictions.
 */
async function fetchPredictions(limit = 60) {
  try {
    const resp = await fetch(`${API_BASE}/api/v1/predictions?limit=${limit}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch {
    return null;
  }
}


// ── UI update functions ───────────────────────────────────────────────────────

function setOnline() {
  els.statusDot.className   = "status-dot online";
  els.statusLabel.textContent = "Live";
}

function setOffline() {
  els.statusDot.className   = "status-dot offline";
  els.statusLabel.textContent = "Disconnected";
}

/** Formats a fractional defect rate as "43.2%" */
function fmt_pct(val)    { return `${(val * 100).toFixed(1)}%`; }
/** Formats milliseconds with one decimal place */
function fmt_ms(val)     { return `${val.toFixed(1)}ms`; }
/** Formats a large integer with comma separators */
function fmt_int(val)    { return val.toLocaleString(); }
/** Formats throughput */
function fmt_thr(val)    { return `${val.toFixed(0)}/min`; }

/**
 * Animates a KPI value update by briefly flashing the value blue.
 */
function animateValue(el, newText) {
  el.textContent = newText;
  el.classList.remove("updated");
  void el.offsetWidth;  // force reflow to restart animation
  el.classList.add("updated");
}

/**
 * Renders a small trend indicator (↑ / ↓) with colour coding.
 * For defect rate and latency, UP is bad (red); for throughput UP is good (green).
 */
function renderTrend(el, current, previous, lowerIsBetter = false) {
  if (previous === null || previous === undefined) {
    el.textContent = "";
    return;
  }
  const delta = current - previous;
  if (Math.abs(delta) < 0.001) { el.textContent = ""; return; }

  const up   = delta > 0;
  const good = lowerIsBetter ? !up : up;

  el.textContent = up ? "↑" : "↓";
  el.style.color = good ? "#10b981" : "#ef4444";
}

/**
 * Updates all four KPI cards with new metric values.
 */
function updateKPIs(metrics) {
  animateValue(els.valTotal,      fmt_int(metrics.total_predictions));
  animateValue(els.valDefect,     fmt_pct(metrics.defect_rate));
  animateValue(els.valLatency,    fmt_ms(metrics.avg_latency_ms));
  animateValue(els.valThroughput, fmt_thr(metrics.throughput_per_min));

  // Trends (compare with previous poll)
  if (prevMetrics) {
    renderTrend(els.trendTotal,      metrics.total_predictions,   prevMetrics.total_predictions, false);
    renderTrend(els.trendDefect,     metrics.defect_rate,         prevMetrics.defect_rate,        true);
    renderTrend(els.trendLatency,    metrics.avg_latency_ms,      prevMetrics.avg_latency_ms,     true);
    renderTrend(els.trendThroughput, metrics.throughput_per_min,  prevMetrics.throughput_per_min, false);
  }

  // Update badges
  els.badgeDefectRate.textContent = fmt_pct(metrics.defect_rate);
  els.badgeP95.textContent        = `p95: ${fmt_ms(metrics.p95_latency_ms)}`;

  // Colour-code the defect badge based on threshold
  if (metrics.defect_rate > 0.5) {
    els.badgeDefectRate.className = "chart-badge chart-badge--red";
  } else {
    els.badgeDefectRate.className = "chart-badge chart-badge--orange";
  }
}

/**
 * Appends the current defect rate to the rolling line chart.
 */
function updateDefectChart(metrics) {
  const now = new Date().toLocaleTimeString("en-GB", { hour12: false });

  defectHistory.push({ x: now, y: metrics.defect_rate });
  if (defectHistory.length > HISTORY_LEN) defectHistory.shift();

  chartDefect.data.labels   = defectHistory.map(d => d.x);
  chartDefect.data.datasets[0].data = defectHistory.map(d => d.y);
  chartDefect.update("none");   // "none" skips animation for smoother live update
}

/**
 * Buckets the latency values from recent predictions into histogram bins.
 */
function updateLatencyChart(predictions) {
  if (!predictions || !predictions.predictions.length) return;

  const buckets = [0, 0, 0, 0, 0, 0, 0];  // <5, 5-10, 10-20, 20-30, 30-50, 50-100, >100
  const thresholds = [5, 10, 20, 30, 50, 100];

  predictions.predictions.forEach(p => {
    const ms = p.latency_ms;
    let idx  = thresholds.findIndex(t => ms < t);
    if (idx === -1) idx = thresholds.length;
    buckets[idx]++;
  });

  chartLatency.data.datasets[0].data = buckets;
  chartLatency.update("none");
}

/**
 * Updates the class distribution donut chart.
 */
function updateDonutChart(metrics) {
  const defective = metrics.defect_rate;
  const normal    = 1 - defective;
  chartDonut.data.datasets[0].data = [
    parseFloat(normal.toFixed(4)),
    parseFloat(defective.toFixed(4)),
  ];
  chartDonut.update("none");

  els.donutCentreVal.textContent = fmt_pct(defective);
}

/**
 * Re-renders the predictions table with the latest rows.
 * The most recent prediction goes at the top.
 */
function updatePredictionsTable(preds) {
  if (!preds || !preds.predictions.length) return;

  const rows = [...preds.predictions].reverse().slice(0, 20);

  els.predTableBody.innerHTML = rows.map((p, i) => {
    const pillClass = p.label === "defective"
      ? "label-pill label-pill--defective"
      : "label-pill label-pill--normal";

    return `
      <tr>
        <td style="color: var(--text-muted); font-family: var(--font-mono)">${i + 1}</td>
        <td><span class="${pillClass}">${p.label}</span></td>
        <td style="font-family: var(--font-mono)">${(p.confidence * 100).toFixed(1)}%</td>
        <td style="font-family: var(--font-mono); color: ${latencyColour(p.latency_ms)}">${p.latency_ms.toFixed(1)}ms</td>
      </tr>
    `;
  }).join("");
}

/** Returns a CSS colour for a latency value. */
function latencyColour(ms) {
  if (ms > 100) return "var(--red)";
  if (ms > 30)  return "var(--orange)";
  return "var(--green)";
}


// ── Polling loop ──────────────────────────────────────────────────────────────

/**
 * One poll cycle: fetch metrics + predictions, update all UI components.
 * Called every POLL_INTERVAL milliseconds.
 */
async function poll() {
  const [metrics, preds] = await Promise.all([
    fetchMetrics(),
    fetchPredictions(60),
  ]);

  if (!metrics) return;

  setOnline();
  updateKPIs(metrics);
  updateDefectChart(metrics);
  updateDonutChart(metrics);
  updateLatencyChart(preds);
  updatePredictionsTable(preds);

  prevMetrics = metrics;
}

function startPolling() {
  if (pollTimerId) return;
  poll();  // immediate first poll
  pollTimerId = setInterval(poll, POLL_INTERVAL);
}

function stopPolling() {
  if (pollTimerId) {
    clearInterval(pollTimerId);
    pollTimerId = null;
  }
}


// ── Upload handler ────────────────────────────────────────────────────────────

async function submitImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  els.uploadResult.hidden = false;
  els.uploadResult.style.background = "rgba(59, 130, 246, 0.08)";
  els.uploadResult.style.border = "1px solid rgba(59, 130, 246, 0.2)";
  els.uploadResult.innerHTML = "<em>Running inference...</em>";

  try {
    const resp = await fetch(`${API_BASE}/api/v1/predict`, {
      method: "POST",
      body:   formData,
    });

    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }

    const isDefective = data.class_idx === 1;
    const bgColor = isDefective
      ? "rgba(239, 68, 68, 0.08)"
      : "rgba(16, 185, 129, 0.08)";
    const borderColor = isDefective
      ? "rgba(239, 68, 68, 0.2)"
      : "rgba(16, 185, 129, 0.2)";
    const labelColor = isDefective ? "var(--red)" : "var(--green)";

    els.uploadResult.style.background = bgColor;
    els.uploadResult.style.border     = `1px solid ${borderColor}`;
    els.uploadResult.innerHTML = `
      <div style="font-weight:600; color:${labelColor}; font-size:1.1rem; margin-bottom:6px">
        ${data.label.toUpperCase()}
      </div>
      <div style="color: var(--text-secondary)">
        Confidence: <strong style="font-family: var(--font-mono)">${(data.confidence * 100).toFixed(2)}%</strong>
      </div>
      <div style="color: var(--text-secondary)">
        Latency: <strong style="font-family: var(--font-mono)">${data.latency_ms.toFixed(1)}ms</strong>
      </div>
      <div style="margin-top:8px; font-size:0.72rem; color: var(--text-muted)">
        Normal: ${(data.probabilities.normal * 100).toFixed(1)}% &nbsp;|&nbsp;
        Defective: ${(data.probabilities.defective * 100).toFixed(1)}%
      </div>
    `;
  } catch (err) {
    els.uploadResult.style.background = "rgba(239, 68, 68, 0.08)";
    els.uploadResult.style.border     = "1px solid rgba(239, 68, 68, 0.2)";
    els.uploadResult.innerHTML = `<span style="color:var(--red)">Error: ${err.message}</span>`;
  }
}


// ── Event listeners ───────────────────────────────────────────────────────────

els.btnTogglePause.addEventListener("click", () => {
  isPaused = !isPaused;
  els.iconPause.style.display = isPaused ? "none"  : "block";
  els.iconPlay.style.display  = isPaused ? "block" : "none";
  isPaused ? stopPolling() : startPolling();
});

els.btnUploadDemo.addEventListener("click", () => {
  els.uploadPanel.hidden = !els.uploadPanel.hidden;
});

els.btnCloseUpload.addEventListener("click", () => {
  els.uploadPanel.hidden = true;
});

// Click on upload area → trigger file input
els.uploadArea.addEventListener("click", () => els.fileInput.click());

// Drag and drop support
els.uploadArea.addEventListener("dragover", e => {
  e.preventDefault();
  els.uploadArea.style.borderColor = "var(--blue)";
  els.uploadArea.style.background  = "var(--glow-blue)";
});

els.uploadArea.addEventListener("dragleave", () => {
  els.uploadArea.style.borderColor = "";
  els.uploadArea.style.background  = "";
});

els.uploadArea.addEventListener("drop", e => {
  e.preventDefault();
  els.uploadArea.style.borderColor = "";
  els.uploadArea.style.background  = "";
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) submitImage(file);
});

els.fileInput.addEventListener("change", () => {
  const file = els.fileInput.files[0];
  if (file) submitImage(file);
  els.fileInput.value = "";  // reset so same file can be re-uploaded
});


// ── Bootstrap ─────────────────────────────────────────────────────────────────

function init() {
  initDefectRateChart();
  initLatencyHistChart();
  initDonutChart();
  startPolling();
}

// Wait for DOM + Chart.js to be ready
document.addEventListener("DOMContentLoaded", init);
