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

function renderShap(res) {
  const section = $("explain");
  if (!Array.isArray(res.shap) || !res.shap.length) { section.hidden = true; return; }
  section.hidden = false;

  const items = [...res.shap].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const biggest = Math.max(...items.map((d) => Math.abs(d.contribution)), 1e-9);

  $("shap-chart").innerHTML = items.map((d) => {
    const f = labelFor(d.feature);
    const side = d.contribution >= 0 ? "pos" : "neg";
    const width = (Math.abs(d.contribution) / biggest) * 50; // half the track = largest bar
    const shown = d.value === null || d.value === undefined ? "left blank (typical value used)" : `${fmt(d.value)} ${f.unit}`;
    return `
      <div class="shap-row">
        <div class="shap-name">${escapeHtml(f.label)}<small>${escapeHtml(shown)}</small></div>
        <div class="shap-track" aria-hidden="true"><div class="shap-bar ${side}" style="width:${width}%"></div></div>
        <div class="shap-num ${side}">${fmtSigned(d.contribution)}</div>
      </div>`;
  }).join("");

  const total = items.reduce((s, d) => s + d.contribution, 0);
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
