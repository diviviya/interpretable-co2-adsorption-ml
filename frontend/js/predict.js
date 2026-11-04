// =====================================================================
// predict.js — Page A (prediction form) + Page B (result and SHAP).
// =====================================================================

renderNav("predict");

let ranges = {};        // min / max / median for each column (from the backend)
let activeModel = null; // version + test metrics of the model in use

// ---------- Build the 10 input boxes from FIELDS ----------
for (const f of FIELDS) {
  const box = document.createElement("div");
  box.className = "field";
  box.id = `field-${f.id}`;
  box.innerHTML = `
    <label for="in-${f.id}">${f.label} <span class="unit">(${f.unit})</span>${f.required ? "" : '<span class="optional">optional</span>'}</label>
    <input id="in-${f.id}" type="number" step="any" inputmode="decimal">
    <small class="hint" id="hint-${f.id}"></small>
    <small class="msg" id="msg-${f.id}"></small>`;
  $(`group-${f.group}`).appendChild(box);
}

// ---------- Load ranges + active model info ----------
async function loadSetup() {
  try {
    ranges = await API.getFeatureInfo();
    for (const f of FIELDS) {
      const r = ranges[f.column];
      if (!r) continue;
      $(`hint-${f.id}`).textContent = `Data range: ${fmt(r.min)} – ${fmt(r.max)}`;
      $(`in-${f.id}`).placeholder = `e.g. ${fmt(r.median)}`;
    }
  } catch (err) {
    showLoadAlert(`Couldn't load the training ranges, so range warnings are off. (${err.message})`);
    $("example-btn").disabled = true;
  }
  try {
    activeModel = await API.getActiveModel();
    $("foot").textContent = `Model in use: ${activeModel.version} · ${activeModel.model_type}`;
  } catch { /* the footer just stays empty */ }
}

function showLoadAlert(text) {
  const box = $("load-alert");
  box.hidden = false;
  box.innerHTML = `<li class="error">${escapeHtml(text)}</li>`;
}

// ---------- Read and check the inputs ----------
function setFieldState(f, state, text) {
  const box = $(`field-${f.id}`);
  box.classList.remove("is-warn", "is-error");
  if (state) box.classList.add(`is-${state}`);
  $(`msg-${f.id}`).textContent = text || "";
}

function readInputs() {
  const features = {}; // keys are the exact dataset column names
  const errors = [];
  const warnings = [];

  for (const f of FIELDS) {
    const raw = $(`in-${f.id}`).value.trim();
    setFieldState(f, null);

    if (raw === "") {
      if (f.required) {
        setFieldState(f, "error", "Required");
        errors.push(`${f.label} is required.`);
      } else {
        features[f.column] = null; // the backend fills blanks with the training median
        setFieldState(f, "warn", "Left blank");
        warnings.push(`${f.label} was left blank, so a typical value is used. The prediction is less reliable.`);
      }
      continue;
    }

    const value = Number(raw);
    if (!f.allowNegative && value < 0) {
      setFieldState(f, "error", "Can't be negative");
      errors.push(`${f.label} can't be negative.`);
      continue;
    }
    if (f.percent && value > 100) {
      setFieldState(f, "error", "Must be 100 or less");
      errors.push(`${f.label} can't be more than 100%.`);
      continue;
    }
    const r = ranges[f.column];
    if (r && (value < r.min || value > r.max)) {
      setFieldState(f, "warn", "Outside data range");
      warnings.push(`${f.label} (${value} ${f.unit}) is outside the training range ${fmt(r.min)}–${fmt(r.max)}. The model is guessing beyond what it has seen.`);
    }
    features[f.column] = value;
  }

  const total = FIELDS.filter((f) => f.percent && features[f.column] != null)
                      .reduce((sum, f) => sum + features[f.column], 0);
  if (total > 100.5) warnings.push(`C + H + N + O + S adds up to ${fmt(total)}%, which is over 100%. Check the composition values.`);

  return { features, errors, warnings };
}

// ---------- Show the prediction ----------
function showAlerts(items) {
  $("result-alerts").innerHTML = items
    .map(([kind, text]) => `<li class="${kind}">${escapeHtml(text)}</li>`).join("");
}

function showPrediction(res, warnings) {
  $("result-empty").hidden = true;
  $("result-body").hidden = false;
  $("mock-badge").hidden = !res.mock;
  $("result-value").textContent = Number(res.prediction).toFixed(2);

  const parts = [`Model version: ${res.model_version}`];
  if (activeModel && activeModel.metrics && activeModel.version === res.model_version) {
    parts.push(`typical error ± ${fmt(activeModel.metrics.rmse, 2)} mmol/g (test RMSE)`);
  }
  $("result-meta").textContent = parts.join(" · ");
  showAlerts([...(res.warnings || []), ...warnings].map((w) => ["", w]));
}

function showError(text, extra = []) {
  $("result-empty").hidden = true;
  $("result-body").hidden = false;
  $("mock-badge").hidden = true;
  $("result-value").textContent = "–";
  $("result-meta").textContent = "";
  showAlerts([["error", text], ...extra.map((e) => ["error", e])]);
  $("explain").hidden = true;
}

// ---------- SHAP chart (only real numbers from the API response) ----------
function labelFor(column) {
  return FIELDS.find((f) => f.column === column) || { label: column, unit: "" };
}

// Waterfall chart: start at the model's average (base value), then each
// input adds or subtracts its contribution, biggest first, until we reach
// the prediction. Bars sit on one shared mmol/g scale, so their lengths
// can be compared directly and the last bar ends exactly at the prediction.
function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((k) => k >= raw) || raw;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(Number(t.toFixed(6)));
  return ticks;
}

function renderShap(res) {
  const section = $("explain");
  if (!Array.isArray(res.shap) || !res.shap.length) { section.hidden = true; return; }
  section.hidden = false;

  const items = [...res.shap].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const base = res.base_value ?? 0;

  // Running total: where each bar starts and ends on the mmol/g scale.
  let running = base;
  const steps = items.map((d) => {
    const start = running;
    running += d.contribution;
    return { ...d, start, end: running };
  });

  const points = [base, res.prediction, ...steps.flatMap((st) => [st.start, st.end])];
  let lo = Math.min(...points), hi = Math.max(...points);
  const pad = (hi - lo) * 0.08 || 0.1;
  lo -= pad; hi += pad;
  const x = (v) => ((v - lo) / (hi - lo)) * 100;

  const rows = steps.map((st) => {
    const f = labelFor(st.feature);
    const side = st.contribution >= 0 ? "pos" : "neg";
    const left = x(Math.min(st.start, st.end));
    const width = Math.abs(x(st.end) - x(st.start));
    const shown = st.value === null || st.value === undefined ? "left blank (typical value used)" : `${fmt(st.value)} ${f.unit}`;
    const tip = `${f.label} = ${shown}: ${st.contribution >= 0 ? "raises" : "lowers"} the prediction by ${fmt(Math.abs(st.contribution))} mmol/g (from ${fmt(st.start)} to ${fmt(st.end)})`;
    return `
      <div class="wf-row" title="${escapeHtml(tip)}">
        <div class="wf-name">${escapeHtml(f.label)}<small>${escapeHtml(shown)}</small></div>
        <div class="wf-track"><span class="wf-bar ${side}" style="left:${left}%;width:${width}%"></span></div>
        <div class="wf-num">${fmtSigned(st.contribution)}</div>
      </div>`;
  }).join("");

  const ticks = niceTicks(lo, hi).map((t) => `<span style="left:${x(t)}%">${fmt(t, 2)}</span>`).join("");

  $("shap-chart").innerHTML = `
    <div class="wf" style="--base:${x(base)}%" role="img" aria-label="Waterfall chart: model average ${fmt(base)} plus each input's contribution gives the prediction ${fmt(res.prediction)} mmol/g. The same numbers are in the table below.">
      <div class="wf-row wf-ref">
        <div class="wf-name">Model average<small>before looking at any input</small></div>
        <div class="wf-track"><span class="wf-tick" style="left:${x(base)}%"></span></div>
        <div class="wf-num">${fmt(base)}</div>
      </div>
      ${rows}
      <div class="wf-row wf-total">
        <div class="wf-name">Prediction</div>
        <div class="wf-track"><span class="wf-tick" style="left:${x(res.prediction)}%"></span></div>
        <div class="wf-num">${fmt(res.prediction)}</div>
      </div>
      <div class="wf-axis"><div></div><div class="wf-axis-track">${ticks}</div><div></div></div>
      <div class="wf-axis"><div class="wf-axis-label">CO₂ uptake (mmol/g)</div></div>
    </div>`;

  // The same numbers as a table, for screen readers and anyone who prefers it.
  $("shap-table").innerHTML = `<table>
    <thead><tr><th>Input</th><th class="num">Value</th><th class="num">Contribution (mmol/g)</th><th class="num">Running total</th></tr></thead>
    <tbody>
      <tr><td>Model average</td><td class="num">–</td><td class="num">–</td><td class="num">${fmt(base)}</td></tr>
      ${steps.map((st) => {
        const f = labelFor(st.feature);
        return `<tr><td>${escapeHtml(f.label)}</td><td class="num">${st.value == null ? "blank" : escapeHtml(fmt(st.value) + " " + f.unit)}</td><td class="num">${fmtSigned(st.contribution)}</td><td class="num">${fmt(st.end)}</td></tr>`;
      }).join("")}
      <tr><td><strong>Prediction</strong></td><td class="num"></td><td class="num"></td><td class="num"><strong>${fmt(res.prediction)}</strong></td></tr>
    </tbody></table>`;

  const total = items.reduce((sum, d) => sum + d.contribution, 0);
  $("shap-sum").innerHTML = res.base_value !== undefined
    ? `Model average <strong>${fmt(res.base_value)}</strong> ${total >= 0 ? "+" : "−"} contributions <strong>${fmt(Math.abs(total))}</strong> = prediction <strong>${fmt(res.prediction)}</strong> mmol/g`
    : "";

  $("shap-text").textContent = res.explanation || summarise(items, res);
}

// Plain-language summary built only from the SHAP numbers the API returned.
function summarise(items, res) {
  const up = items.filter((d) => d.contribution > 0.0005).slice(0, 2);
  const down = items.filter((d) => d.contribution < -0.0005).slice(0, 2);
  const names = (list) => list.map((d) => `${labelFor(d.feature).label.toLowerCase()} (${fmtSigned(d.contribution)})`).join(" and ");
  const sentences = [];
  if (up.length) sentences.push(`The biggest push upward came from ${names(up)}.`);
  if (down.length) sentences.push(`The biggest push downward came from ${names(down)}.`);
  if (!sentences.length) sentences.push("No input moved this prediction much away from the model's average.");
  if (res.base_value !== undefined) {
    sentences.push(`Overall the prediction is ${res.prediction >= res.base_value ? "above" : "below"} the model's average of ${fmt(res.base_value, 2)} mmol/g.`);
  }
  return sentences.join(" ");
}

// ---------- Buttons ----------
$("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const { features, errors, warnings } = readInputs();
  if (errors.length) {
    showError(errors[0], errors.slice(1));
    document.querySelector(".field.is-error input")?.focus();
    return;
  }

  const btn = $("predict-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';
  try {
    const res = await API.predict(features);
    showPrediction(res, warnings);
    renderShap(res);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict uptake";
  }
});

$("example-btn").addEventListener("click", () => {
  for (const f of FIELDS) {
    const r = ranges[f.column];
    if (r) $(`in-${f.id}`).value = fmt(r.median);
  }
});

$("predict-form").addEventListener("reset", () => {
  for (const f of FIELDS) setFieldState(f, null);
  $("result-empty").hidden = false;
  $("result-body").hidden = true;
  $("explain").hidden = true;
});

loadSetup();
