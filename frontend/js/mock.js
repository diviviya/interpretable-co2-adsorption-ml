// =====================================================================
// mock.js — FAKE backend responses so the site works before FastAPI exists.
//
// Every response here has the same SHAPE the real backend will return,
// so switching CONFIG.USE_MOCK to false needs no other change.
//
// What is real and what is not:
//   REAL : feature ranges (artifacts/feature_info.json), global SHAP
//          importance (results/08_shap_importance.csv), and the active
//          model's test metrics (Notebook 07).
//   FAKE : every individual prediction and its per-feature contributions,
//          the training job, and the new candidate model's metrics.
// =====================================================================

const MOCK = (() => {
  const wait = (ms = 400) => new Promise((r) => setTimeout(r, ms));

  // ---- REAL numbers from the notebooks ----
  const ACTIVE_MODEL = {
    version: "v1-ensemble",
    created_at: "2026-10-28T15:15:00+05:30",
    model_type: "Entropy-TOPSIS weighted ensemble of 6 tree models",
    status: "active",
    dataset: "Yuan et al. (2021) biomass carbons, 526 rows",
    metrics: { rmse: 0.644, mae: 0.477, r2: 0.848 }, // 106 held-out test rows
    metrics_split: "test (106 rows)",
  };
  const RIDGE_BASELINE = {
    version: "v0-ridge",
    created_at: "2026-10-08T14:01:00+05:30",
    model_type: "Ridge regression baseline",
    status: "previous",
    dataset: "Yuan et al. (2021) biomass carbons, 526 rows",
    metrics: { rmse: 0.974, mae: 0.745, r2: 0.65 },
    metrics_split: "test (106 rows)",
  };

  // ---- FAKE "model": a straight-line guess around the training medians ----
  const BASE = 2.6;
  const SLOPE = {
    "Surface Area (m2/g)": 0.0006, "Total Pore Volume(cm3/g)": 0.3, "Micropore Volume (cm3/g)": 0.8,
    "C (%)": 0.01, "H (%)": -0.05, "N (%)": 0.04, "O (%)": -0.01, "S (%)": 0,
    "Temp (°C)": -0.04, "Pressure (bar)": 1.5,
  };

  let featureInfoCache = null;
  async function getFeatureInfo() {
    if (featureInfoCache) return featureInfoCache;
    const res = await fetch(CONFIG.MOCK_FEATURE_INFO_URL);
    if (!res.ok) throw new Error("Couldn't load artifacts/feature_info.json. Run the site from the repo folder (see README).");
    featureInfoCache = await res.json();
    return featureInfoCache;
  }

  function fakeExplain(info, features) {
    const shap = FEATURE_COLUMNS.map((c) => {
      const v = features[c];
      const median = info[c] ? info[c].median : 0;
      const x = v === null || v === undefined ? median : v;
      return { feature: c, value: v ?? null, contribution: Number((SLOPE[c] * (x - median)).toFixed(4)) };
    });
    const total = shap.reduce((s, d) => s + d.contribution, 0);
    return { prediction: Math.max(BASE + total, 0.05), shap };
  }

  // Mock only: remember predictions for the History page within this tab.
  // (The real History page fetches from the backend, per the brief.)
  const HISTORY_KEY = "mock-history";
  function readHistory() {
    try { return JSON.parse(sessionStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
  }
  function saveHistory(items) {
    try { sessionStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 50))); } catch { /* storage blocked: fine */ }
  }

  async function predict(features) {
    const info = await getFeatureInfo();
    await wait(500);
    const { prediction, shap } = fakeExplain(info, features);
    const result = {
      prediction: Number(prediction.toFixed(3)),
      unit: "mmol/g",
      model_version: "mock-demo",
      base_value: BASE,
      shap,
      explanation: null, // the real backend may send its own sentence here
      mock: true,
    };
    const items = readHistory();
    items.unshift({ timestamp: new Date().toISOString(), model_version: result.model_version, prediction: result.prediction, features });
    saveHistory(items);
    return result;
  }

  async function predictBatch(file) {
    const info = await getFeatureInfo();
    const { columns, rows } = parseCSV(await file.text());
    await wait(700);

    const missing = FEATURE_COLUMNS.filter((c) => !columns.includes(c));
    if (missing.length) throw new Error("Missing columns: " + missing.join(", "));

    const errors = [];
    const out = [];
    rows.forEach((r, i) => {
      const features = {};
      for (const c of FEATURE_COLUMNS) {
        const raw = r[c];
        if (raw === "") { features[c] = null; continue; }
        const v = Number(raw);
        if (Number.isNaN(v)) { errors.push({ row: i + 2, message: `"${c}" is not a number ("${raw}")` }); return; }
        features[c] = v;
      }
      out.push({ row: i + 2, ...features, prediction: Number(fakeExplain(info, features).prediction.toFixed(3)) });
    });
    return { model_version: "mock-demo", rows: out, errors, mock: true };
  }

  async function getHistory() {
    await wait(300);
    return { items: readHistory(), mock: true };
  }

  async function getImportance() {
    const res = await fetch(CONFIG.MOCK_IMPORTANCE_URL);
    if (!res.ok) throw new Error("Couldn't load results/08_shap_importance.csv.");
    const { columns, rows } = parseCSV(await res.text());
    const [nameCol, valueCol] = columns;
    return {
      model_version: ACTIVE_MODEL.version,
      importance: rows.map((r) => ({ feature: r[nameCol], mean_abs_shap: Number(r[valueCol]) })),
    };
  }

  // ---- admin side ----
  let models = [structuredClone(ACTIVE_MODEL), structuredClone(RIDGE_BASELINE)];
  const jobs = {};

  async function uploadDataset(file) {
    const { columns, rows } = parseCSV(await file.text());
    await wait(600);
    const required = [...FEATURE_COLUMNS, TARGET];
    const missingCols = required.filter((c) => !columns.includes(c));
    const missing_values = {};
    for (const c of columns) missing_values[c] = rows.filter((r) => r[c] === "").length;
    const seen = new Set();
    let duplicate_rows = 0;
    for (const r of rows) {
      const key = required.map((c) => r[c]).join("|");
      if (seen.has(key)) duplicate_rows++; else seen.add(key);
    }
    return {
      dataset_id: "ds-" + Date.now(),
      filename: file.name,
      rows: rows.length,
      columns,
      missing_values,
      duplicate_rows,
      errors: missingCols.length ? ["Missing required columns: " + missingCols.join(", ")] : [],
      mock: true,
    };
  }

  async function startTraining(datasetId) {
    await wait(300);
    const id = "job-" + Date.now();
    jobs[id] = { started: Date.now(), datasetId };
    return { job_id: id, mock: true };
  }

  const STAGES = [
    [0.00, "Loading dataset and checking columns"],
    [0.10, "Removing duplicates, splitting 64/16/20 with seed 42"],
    [0.25, "Refitting 6 models with saved best params"],
    [0.70, "Computing entropy-TOPSIS ensemble weights"],
    [0.85, "Evaluating candidate on held-out test rows"],
    [0.95, "Saving candidate model"],
  ];

  async function getTrainingStatus(jobId) {
    await wait(200);
    const job = jobs[jobId];
    if (!job) throw new Error("Unknown training job.");
    const progress = Math.min((Date.now() - job.started) / 9000, 1);
    const logs = STAGES.filter(([p]) => progress >= p).map(([, msg]) => msg);
    if (progress < 1) return { job_id: jobId, status: "running", progress, logs, mock: true };

    const version = "v2-candidate";
    if (!models.some((m) => m.version === version)) {
      models.unshift({
        version, created_at: new Date().toISOString(),
        model_type: ACTIVE_MODEL.model_type, status: "candidate",
        dataset: "uploaded dataset " + job.datasetId,
        metrics: { rmse: 0.63, mae: 0.47, r2: 0.85 }, // FAKE demo numbers
        metrics_split: "test",
      });
    }
    return {
      job_id: jobId, status: "succeeded", progress: 1,
      logs: [...logs, "Done: candidate " + version + " ready for review"],
      candidate: structuredClone(models.find((m) => m.version === version)),
      mock: true,
    };
  }

  async function listModels() {
    await wait(300);
    return { models: structuredClone(models), mock: true };
  }

  function makeActive(version) {
    const target = models.find((m) => m.version === version);
    if (!target) throw new Error("Unknown version " + version);
    models.forEach((m) => { if (m.status === "active") m.status = "previous"; });
    target.status = "active";
    return { models: structuredClone(models), mock: true };
  }

  async function activateModel(version) { await wait(400); return makeActive(version); }
  async function rollbackModel(version) { await wait(400); return makeActive(version); }

  // Demo mode accepts any username and password. The real backend checks them.
  async function login(username, password, admin) {
    await wait(400);
    if (!username || !password) throw new Error("Enter a username and password.");
    return { token: "mock-token", user: { id: 1, username, role: admin ? "admin" : "user" }, mock: true };
  }
  async function signup(username, password) { return login(username, password, false); }
  async function me() { await wait(100); return { user: Auth.user() }; }
  async function logout() { return { ok: true }; }

  // ---- users (demo) ----
  const users = [
    { id: 1, username: "adhithya", role: "admin", created_at: "2026-11-04T21:00:00+05:30", predictions: 12, main_admin: true },
    { id: 2, username: "divya", role: "user", created_at: "2026-11-05T12:40:00+05:30", predictions: 3, main_admin: false },
    { id: 3, username: "vignesh", role: "user", created_at: "2026-11-05T12:45:00+05:30", predictions: 5, main_admin: false },
  ];
  async function listUsers() { await wait(250); return { users: structuredClone(users), mock: true }; }
  async function setUserRole(userId, role) {
    await wait(300);
    const u = users.find((x) => x.id === Number(userId));
    if (!u) throw new Error("No such user.");
    if (role === "user" && u.main_admin) throw new Error("The main admin (set in backend/.env) always stays an admin.");
    u.role = role;
    return { users: structuredClone(users), mock: true };
  }

  async function health() { return { status: "ok", mock: true }; }
  async function getActiveModel() {
    await wait(200);
    return structuredClone(models.find((m) => m.status === "active"));
  }

  return {
    health, getFeatureInfo, getActiveModel, getImportance, getHistory,
    predict, predictBatch,
    login, signup, me, logout, listUsers, setUserRole, uploadDataset, startTraining, getTrainingStatus, listModels, activateModel, rollbackModel,
  };
})();
