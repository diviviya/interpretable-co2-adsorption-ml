// =====================================================================
// admin.js — dataset upload, training, and model version management.
// Only reachable after logging in as an admin. Every request carries the
// session token, and the BACKEND rejects anyone who isn't an admin.
// =====================================================================

renderNav("admin");
$("target-name").textContent = TARGET;

const REQUIRED = [...FEATURE_COLUMNS, TARGET];
let chosenFile = null;
let validated = null;   // server's validation result for the uploaded dataset
let pollTimer = null;   // repeating check on the running training job

// ---------- Helpers ----------
// The admin is already logged in (admin-login.html); the session token is
// sent automatically by api.js, and the backend checks the admin role.
function alerts(id, items) {
  $(id).innerHTML = items.map(([kind, t]) => `<li class="${kind}">${escapeHtml(t)}</li>`).join("");
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
  if (!chosenFile) return;
  const btn = $("upload-btn");
  busy(btn, "Uploading…");
  try {
    const res = await API.uploadDataset(chosenFile);
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
  if (!validated) return;
  const sure = confirm(`Start training on "${validated.filename}" (${validated.rows} rows)?\n\nThis can take several minutes. The active model keeps serving users until you activate the new one.`);
  if (!sure) return;

  const btn = $("train-btn");
  busy(btn, "Starting…");
  $("train-status").hidden = false;
  $("compare").hidden = true;
  alerts("train-alerts", []);
  try {
    const { job_id } = await API.startTraining(validated.dataset_id);
    $("train-job").textContent = `Job ${job_id}`;
    poll(job_id);
  } catch (err) {
    alerts("train-alerts", [["error", err.message]]);
    idle(btn);
  }
});

function poll(jobId) {
  clearTimeout(pollTimer);
  let misses = 0;   // progress checks in a row that got no reply
  const tick = async () => {
    try {
      const s = await API.getTrainingStatus(jobId);
      misses = 0;
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
      // The server can be slow to answer while it trains (heavy steps keep
      // it busy). One slow reply doesn't mean the job died: keep checking,
      // and only give up after many misses in a row (about 1 minute).
      misses += 1;
      if (misses < 12) {
        $("train-state").textContent = "Training… (server busy, still checking)";
        pollTimer = setTimeout(tick, 5000);
        return;
      }
      idle($("train-btn"));
      alerts("train-alerts", [["error", "Lost contact with the training job: " + err.message + " Click Refresh in section 3 to see if it finished."]]);
    }
  };
  tick();
}

async function showComparison(candidate, isMock) {
  let active = null;
  try {
    const { models } = await API.listModels();
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
    const { models } = await API.listModels();
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
  if (!btn) return;
  const version = btn.dataset.v;
  const action = btn.dataset.act;
  const message = action === "activate"
    ? `Activate ${version}?\n\nEvery prediction on the site will use this model from now on. The current active model becomes "previous" and can be restored later.`
    : `Roll back to ${version}?\n\nEvery prediction on the site will use this older model from now on.`;
  if (!confirm(message)) return;

  busy(btn, action === "activate" ? "Activating…" : "Rolling back…");
  try {
    const res = action === "activate" ? await API.activateModel(version) : await API.rollbackModel(version);
    if (res && res.models) renderVersions(res.models); else await loadVersions();
    alerts("versions-alerts", [["ok", `${version} is now the active model.`]]);
  } catch (err) {
    idle(btn);
    alerts("versions-alerts", [["error", err.message]]);
  }
});

$("refresh-btn").addEventListener("click", loadVersions);

// ---------- 4. Users and admins ----------
async function loadUsers() {
  alerts("users-alerts", []);
  try {
    const { users } = await API.listUsers();
    renderUsers(users);
  } catch (err) {
    $("users-table").innerHTML = "";
    alerts("users-alerts", [["error", err.message]]);
  }
}

function renderUsers(users) {
  const me = Auth.user() || {};
  $("users-table").innerHTML = `<table>
    <thead><tr><th>Username</th><th>Role</th><th>Joined</th><th class="num">Predictions</th><th></th></tr></thead>
    <tbody>${users.map((u) => {
      const isMe = u.id === me.id || u.username === me.username;
      let action = "";
      if (isMe) action = '<span class="muted" style="margin:0">This is you</span>';
      else if (u.main_admin) action = '<span class="muted" style="margin:0">Main admin</span>';
      else if (u.role === "admin") action = `<button class="btn danger small" data-uid="${u.id}" data-role="user" data-name="${escapeHtml(u.username)}">Remove admin</button>`;
      else action = `<button class="btn primary small" data-uid="${u.id}" data-role="admin" data-name="${escapeHtml(u.username)}">Make admin</button>`;
      return `<tr>
        <td><strong>${escapeHtml(u.username)}</strong></td>
        <td><span class="badge ${u.role === "admin" ? "active" : "previous"}">${u.role}</span></td>
        <td>${escapeHtml(fmtDate(u.created_at))}</td>
        <td class="num">${u.predictions ?? "–"}</td>
        <td>${action}</td>
      </tr>`;
    }).join("")}</tbody></table>`;
}

$("users-table").addEventListener("click", async (e) => {
  const btn = e.target.closest("button[data-uid]");
  if (!btn) return;
  const { uid, role, name } = btn.dataset;
  const message = role === "admin"
    ? `Make ${name} an admin?\n\nThey will be able to upload datasets, train models and change which model everyone uses.`
    : `Remove ${name}'s admin role?\n\nThey keep their account and history, but lose access to this page.`;
  if (!confirm(message)) return;

  busy(btn, role === "admin" ? "Saving…" : "Removing…");
  try {
    const { users } = await API.setUserRole(uid, role);
    renderUsers(users);
    alerts("users-alerts", [["ok", role === "admin" ? `${name} is now an admin.` : `${name} is no longer an admin.`]]);
  } catch (err) {
    idle(btn);
    alerts("users-alerts", [["error", err.message]]);
  }
});

$("users-refresh-btn").addEventListener("click", loadUsers);

$("logout-btn").addEventListener("click", logout);
loadVersions();
loadUsers();
