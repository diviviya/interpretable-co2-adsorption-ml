// =====================================================================
// admin.js — dataset upload, training, and model version management.
// Every admin request carries the admin key; the BACKEND must reject
// requests without a valid key. This page only makes the job convenient.
// =====================================================================

renderNav("admin");
$("target-name").textContent = TARGET;

const REQUIRED = [...FEATURE_COLUMNS, TARGET];
let chosenFile = null;
let validated = null;   // server's validation result for the uploaded dataset
let pollTimer = null;   // repeating check on the running training job

// ---------- Login ----------
// The key lives only in this variable: it is gone when the page closes.
let adminKey = "";
const key = () => adminKey;

if (CONFIG.USE_MOCK) {
  $("login-sub").textContent = "Demo mode: any key is accepted. With the real backend, the key is checked by the server and you'll need to log in again after leaving this page.";
}

function alerts(id, items) {
  $(id).innerHTML = items.map(([kind, t]) => `<li class="${kind}">${escapeHtml(t)}</li>`).join("");
}

function showTools(loggedIn) {
  $("login-card").hidden = loggedIn;
  $("admin-tools").hidden = !loggedIn;
}

function logout(message) {
  adminKey = "";
  clearTimeout(pollTimer);
  $("admin-key").value = "";
  showTools(false);
  alerts("login-alerts", message ? [["error", message]] : []);
  $("admin-key").focus();
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const typed = $("admin-key").value.trim();
  if (!typed) { alerts("login-alerts", [["error", "Enter the admin key."]]); return; }
  const btn = $("login-btn");
  busy(btn, "Checking…");
  try {
    await API.adminLogin(typed);   // the backend decides whether the key is valid
    adminKey = typed;
    alerts("login-alerts", []);
    showTools(true);
    loadVersions();
  } catch (err) {
    alerts("login-alerts", [["error", /authoris/i.test(err.message) ? "Invalid admin key." : err.message]]);
  } finally {
    idle(btn);
  }
});

$("logout-btn").addEventListener("click", () => logout());

// If the backend rejects the key later (e.g. it was changed), go back to login.
API.onAuthError(() => logout("Your admin session was rejected by the server. Please log in again."));

// Admin actions only run while logged in.
function needKey() {
  return !adminKey;
}

function busy(btn, label) {
  btn.disabled = true;
  btn.dataset.label = btn.textContent;
  btn.innerHTML = `<span class="spinner"></span> ${label}`;
}
function idle(btn) {
  btn.disabled = false;
  btn.textContent = btn.dataset.label;
}

// ---------- 1. Dataset: local checks + preview ----------
async function handleFile(file) {
  chosenFile = null;
  validated = null;
  $("local-preview").hidden = true;
  $("validation").hidden = true;
  updateTrainButton();
  $("file-name").textContent = file ? file.name : "";

  const problem = checkCsvFile(file);
  if (problem) { alerts("file-alerts", [["error", problem]]); return; }

  let parsed;
  try { parsed = parseCSV(await file.text()); }
  catch { alerts("file-alerts", [["error", "Couldn't read this file as text."]]); return; }

  const missing = REQUIRED.filter((c) => !parsed.columns.includes(c));
  const notes = [[missing.length ? "error" : "ok",
    missing.length ? "Missing columns (names must match exactly): " + missing.join(", ")
                   : `${parsed.rows.length} rows and all ${REQUIRED.length} required columns found.`]];
  alerts("file-alerts", notes);
  if (missing.length || !parsed.rows.length) return;

  chosenFile = file;
  const shown = parsed.rows.slice(0, 8);
  $("preview-sub").textContent = `Preview: first ${shown.length} of ${parsed.rows.length} rows.`;
  $("preview-table").innerHTML = `<table><thead><tr>${REQUIRED.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
    <tbody>${shown.map((r) => `<tr>${REQUIRED.map((c) => `<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("local-preview").hidden = false;
}

$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("is-over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("is-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

// ---------- 1b. Upload and show the server's validation ----------
$("upload-btn").addEventListener("click", async () => {
  if (!chosenFile || needKey()) return;
  const btn = $("upload-btn");
  busy(btn, "Uploading…");
  try {
    const res = await API.uploadDataset(chosenFile, key());
    validated = res;
    showValidation(res);
  } catch (err) {
    validated = null;
    $("validation").hidden = false;
    $("val-stats").innerHTML = "";
    $("missing-table").hidden = true;
    alerts("val-alerts", [["error", err.message]]);
  } finally {
    idle(btn);
    updateTrainButton();
  }
});

function showValidation(res) {
  $("validation").hidden = false;
  $("val-mock").hidden = !res.mock;
  const missingTotal = Object.values(res.missing_values || {}).reduce((a, b) => a + b, 0);
  $("val-stats").innerHTML = [
    ["Rows", res.rows],
    ["Columns", (res.columns || []).length],
    ["Missing cells", missingTotal],
    ["Duplicate rows", res.duplicate_rows],
  ].map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${escapeHtml(v)}</div></div>`).join("");

  const items = (res.errors || []).map((e) => ["error", e]);
  if (res.duplicate_rows) items.push(["", `${res.duplicate_rows} duplicate row${res.duplicate_rows === 1 ? "" : "s"} found. The pipeline removes duplicates before training.`]);
  if (missingTotal) items.push(["", `${missingTotal} missing values. The pipeline fills them with the training median. Columns with many gaps carry weaker information.`]);
  if (!items.length || !(res.errors || []).length) items.push(["ok", "Dataset accepted. You can start training."]);
  alerts("val-alerts", items);

  const gaps = Object.entries(res.missing_values || {}).filter(([, n]) => n > 0);
  $("missing-table").hidden = !gaps.length;
  $("missing-table").innerHTML = gaps.length ? `<table>
    <thead><tr><th>Column</th><th class="num">Missing</th><th class="num">Share</th></tr></thead>
    <tbody>${gaps.map(([c, n]) => `<tr><td>${escapeHtml(c)}</td><td class="num">${n}</td><td class="num">${fmt((n / res.rows) * 100, 1)}%</td></tr>`).join("")}</tbody></table>` : "";
}

// ---------- 2. Training ----------
function updateTrainButton() {
  const ok = validated && !(validated.errors || []).length;
  $("train-btn").disabled = !ok;
  $("train-hint").hidden = !!ok;
}

$("train-btn").addEventListener("click", async () => {
  if (!validated || needKey()) return;
  const sure = confirm(`Start training on "${validated.filename}" (${validated.rows} rows)?\n\nThis can take several minutes. The active model keeps serving users until you activate the new one.`);
  if (!sure) return;

  const btn = $("train-btn");
  busy(btn, "Starting…");
  $("train-status").hidden = false;
  $("compare").hidden = true;
  alerts("train-alerts", []);
  try {
    const { job_id } = await API.startTraining(validated.dataset_id, key());
    $("train-job").textContent = `Job ${job_id}`;
    poll(job_id);
  } catch (err) {
    alerts("train-alerts", [["error", err.message]]);
    idle(btn);
  }
});

function poll(jobId) {
  clearTimeout(pollTimer);
  const tick = async () => {
    try {
      const s = await API.getTrainingStatus(jobId, key());
      $("train-state").textContent = { queued: "Queued", running: "Training…", succeeded: "Training finished", failed: "Training failed" }[s.status] || s.status;
      $("train-bar").style.width = `${Math.round((s.progress ?? 0) * 100)}%`;
      $("train-logs").textContent = (s.logs || []).join("\n");
      $("train-logs").scrollTop = $("train-logs").scrollHeight;

      if (s.status === "succeeded") {
        idle($("train-btn"));
        alerts("train-alerts", [["ok", `Candidate ${s.candidate ? s.candidate.version : ""} is ready. Review it below, then activate it in section 3 if it is better.`]]);
        if (s.candidate) showComparison(s.candidate, !!s.mock);
        loadVersions();
        return;
      }
      if (s.status === "failed") {
        idle($("train-btn"));
        alerts("train-alerts", [["error", s.error || "Training failed. Check the logs above."]]);
        return;
      }
      pollTimer = setTimeout(tick, 1500);
    } catch (err) {
      idle($("train-btn"));
      alerts("train-alerts", [["error", "Lost contact with the training job: " + err.message]]);
    }
  };
  tick();
}

async function showComparison(candidate, isMock) {
  let active = null;
  try {
    const { models } = await API.listModels(key());
    active = models.find((m) => m.status === "active");
  } catch { /* show the candidate alone */ }

  const rows = [["RMSE (mmol/g)", "rmse", true], ["MAE (mmol/g)", "mae", true], ["R²", "r2", false]];
  const better = (c, a, lowerIsBetter) => (a == null ? "" : (lowerIsBetter ? c < a : c > a) ? "Candidate" : c === a ? "Same" : "Active");
  $("compare").hidden = false;
  $("cmp-mock").hidden = !isMock;
  $("compare-table").innerHTML = `<table>
    <thead><tr><th>Metric</th><th class="num">Candidate ${escapeHtml(candidate.version)}</th><th class="num">Active ${escapeHtml(active ? active.version : "–")}</th><th>Better</th></tr></thead>
    <tbody>${rows.map(([label, k, low]) => {
      const c = candidate.metrics[k];
      const a = active ? active.metrics[k] : null;
      return `<tr><td>${label}</td><td class="num">${fmt(c)}</td><td class="num">${fmt(a)}</td><td>${better(c, a, low)}</td></tr>`;
    }).join("")}</tbody></table>`;
}

// ---------- 3. Versions ----------
async function loadVersions() {
  alerts("versions-alerts", []);
  try {
    const { models } = await API.listModels(key());
    renderVersions(models);
  } catch (err) {
    $("versions-table").innerHTML = "";
    alerts("versions-alerts", [["error", err.message]]);
  }
}

function renderVersions(models) {
  if (!models.length) { $("versions-table").innerHTML = '<p class="muted" style="padding:12px">No model versions yet.</p>'; return; }
  $("versions-table").innerHTML = `<table>
    <thead><tr><th>Version</th><th>Status</th><th>Created</th><th>Model</th><th class="num">RMSE</th><th class="num">MAE</th><th class="num">R²</th><th>Training data</th><th></th></tr></thead>
    <tbody>${models.map((m) => `
      <tr>
        <td><code>${escapeHtml(m.version)}</code></td>
        <td><span class="badge ${escapeHtml(m.status)}">${escapeHtml(m.status)}</span></td>
        <td>${escapeHtml(fmtDate(m.created_at))}</td>
        <td class="wrap-text">${escapeHtml(m.model_type)}</td>
        <td class="num">${fmt(m.metrics && m.metrics.rmse)}</td>
        <td class="num">${fmt(m.metrics && m.metrics.mae)}</td>
        <td class="num">${fmt(m.metrics && m.metrics.r2)}</td>
        <td class="wrap-text">${escapeHtml(m.dataset)}</td>
        <td>${m.status === "candidate" ? `<button class="btn primary small" data-act="activate" data-v="${escapeHtml(m.version)}">Activate</button>`
             : m.status === "previous" ? `<button class="btn danger small" data-act="rollback" data-v="${escapeHtml(m.version)}">Roll back to this</button>`
             : ""}</td>
      </tr>`).join("")}</tbody></table>`;
}

$("versions-table").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn || needKey()) return;
  const version = btn.dataset.v;
  const action = btn.dataset.act;
  const message = action === "activate"
    ? `Activate ${version}?\n\nEvery prediction on the site will use this model from now on. The current active model becomes "previous" and can be restored later.`
    : `Roll back to ${version}?\n\nEvery prediction on the site will use this older model from now on.`;
  if (!confirm(message)) return;

  busy(btn, action === "activate" ? "Activating…" : "Rolling back…");
  try {
    const res = action === "activate" ? await API.activateModel(version, key()) : await API.rollbackModel(version, key());
    if (res && res.models) renderVersions(res.models); else await loadVersions();
    alerts("versions-alerts", [["ok", `${version} is now the active model.`]]);
  } catch (err) {
    idle(btn);
    alerts("versions-alerts", [["error", err.message]]);
  }
});

$("refresh-btn").addEventListener("click", () => { if (!needKey()) loadVersions(); });

// Start on the login screen.
showTools(false);
$("admin-key").focus();
