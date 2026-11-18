// =====================================================================
// about.js — Page E: fills the About page with live numbers from the API.
// Nothing here is typed in by hand: metrics, ranges and importance all
// come from the backend (or, in demo mode, from the project's own files).
// =====================================================================

renderNav("about");
$("target-name").textContent = TARGET;

async function loadFeatures() {
  let info = {};
  try { info = await API.getFeatureInfo(); } catch { /* table still shows names and units */ }
  const rows = FIELDS.map((f) => {
    const r = info[f.column];
    return `<tr>
      <td>${escapeHtml(f.label)}</td>
      <td><code>${escapeHtml(f.column)}</code></td>
      <td>${escapeHtml(f.unit)}</td>
      <td class="num">${r ? fmt(r.min) : "–"}</td>
      <td class="num">${r ? fmt(r.max) : "–"}</td>
      <td>${f.required ? "Required" : "Optional"}</td>
    </tr>`;
  }).join("");
  $("features-table").innerHTML = `<table>
    <thead><tr><th>Input</th><th>Column name</th><th>Unit</th><th class="num">Min</th><th class="num">Max</th><th>In the form</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

async function loadMetrics() {
  try {
    const m = await API.getActiveModel();
    $("metrics-sub").textContent = `Model ${m.version} (${m.model_type}). Measured on the ${m.metrics_split || "held-out test"} rows, which the model never saw during training or tuning.`;
    const k = m.metrics || {};
    $("metrics").innerHTML = [
      ["RMSE", k.rmse, "mmol/g"],
      ["MAE", k.mae, "mmol/g"],
      ["R²", k.r2, "variation explained"],
    ].map(([name, v, d]) => `<div class="stat"><div class="k">${name}</div><div class="v">${fmt(v)}</div><div class="d">${d}</div></div>`).join("");
  } catch (err) {
    $("metrics-sub").textContent = "Couldn't load the model's metrics: " + err.message;
  }
}

async function loadImportance() {
  try {
    const res = await API.getImportance();
    const items = [...res.importance].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);
    const top = Math.max(...items.map((d) => d.mean_abs_shap), 1e-9);
    $("importance").innerHTML = items.map((d) => {
      const f = FIELDS.find((x) => x.column === d.feature);
      return `<div class="imp-row">
        <div class="shap-name">${escapeHtml(f ? f.label : d.feature)}</div>
        <div class="imp-track" aria-hidden="true"><div class="imp-bar" style="width:${(d.mean_abs_shap / top) * 100}%"></div></div>
        <div class="shap-num">${fmt(d.mean_abs_shap)}</div>
      </div>`;
    }).join("");
  } catch (err) {
    $("importance-alerts").innerHTML = `<li class="error">${escapeHtml(err.message)}</li>`;
  }
}

loadFeatures();
loadMetrics();
loadImportance();
