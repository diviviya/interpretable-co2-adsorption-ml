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
    [0.25, "Tuning 6 models with Optuna (5-fold CV)"],
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

  // Demo mode accepts any non-empty key. The real backend must check it.
  async function adminLogin(key) {
    await wait(400);
    if (!key) throw new Error("Enter the admin key.");
    return { ok: true, role: "admin", mock: true };
  }

  async function health() { return { status: "ok", mock: true }; }
  async function getActiveModel() {
    await wait(200);
    return structuredClone(models.find((m) => m.status === "active"));
  }


  // ---- BIOMASS SELECTION GATE MOCK ----
  const BIOMASS_PRECURSORS = [
    {
      id: "coconut-shell",
      name: "Coconut Shell Activated Carbon",
      class: "High Microporosity",
      surface_area: 2150.0,
      total_pore_volume: 0.95,
      micropore_volume: 0.72,
      carbon: 88.5,
      hydrogen: 1.8,
      nitrogen: 0.6,
      oxygen: 8.8,
      sulfur: 0.3,
      synthesis_recipe: "Chemical activation with KOH at 800°C (1:3 ratio). Optimal for ambient & high-pressure capture.",
      advantages: [
        "Ultra-high micropore fraction (76%)",
        "Dense slit-pore confinement for CO2",
        "Abundant commercial bio-waste availability"
      ]
    },
    {
      id: "walnut-shell",
      name: "Walnut Shell Activated Carbon",
      class: "Dense High-Hardness Nut Shell",
      surface_area: 1980.0,
      total_pore_volume: 0.88,
      micropore_volume: 0.68,
      carbon: 86.5,
      hydrogen: 1.9,
      nitrogen: 0.5,
      oxygen: 10.7,
      sulfur: 0.4,
      synthesis_recipe: "Two-stage activation: carbonization at 500°C followed by KOH chemical activation at 800°C (1:2.5 ratio).",
      advantages: [
        "Superior mechanical crush and attrition strength",
        "Ultra-high micropore fraction (77%)",
        "Dense slit-pore molecular sieving for CO2"
      ]
    },
    {
      id: "bamboo-biochar",
      name: "Vascular Bamboo Biochar",
      class: "Vascular / High Capacity",
      surface_area: 1750.0,
      total_pore_volume: 0.79,
      micropore_volume: 0.61,
      carbon: 85.0,
      hydrogen: 2.0,
      nitrogen: 0.8,
      oxygen: 11.8,
      sulfur: 0.4,
      synthesis_recipe: "Steam gasification at 750°C of mature bamboo stalks. Preserves natural axial honeycomb vessels.",
      advantages: [
        "Highest overall CO2 capture capacity (4.33 mmol/g)",
        "Open tubular macro-channels for rapid mass transfer kinetics",
        "Rapidly renewable vegetative biomass"
      ]
    },
    {
      id: "sugarcane-bagasse",
      name: "Sugarcane Bagasse Biochar",
      class: "Agro-Industrial Fibrous",
      surface_area: 1620.0,
      total_pore_volume: 0.82,
      micropore_volume: 0.58,
      carbon: 81.0,
      hydrogen: 2.1,
      nitrogen: 0.8,
      oxygen: 15.5,
      sulfur: 0.6,
      synthesis_recipe: "De-ashing acid wash followed by KOH chemical activation at 750°C. Upcycles sugar milling byproducts.",
      advantages: [
        "Massive industrial agro-waste availability from sugar mills",
        "High fibrous surface area (1620 m²/g)",
        "Balanced meso-microporosity with low transport resistance"
      ]
    },
    {
      id: "chitosan-biochar",
      name: "Nitrogen-Enriched Chitosan Biochar",
      class: "Heteroatom / Chemical Affinity",
      surface_area: 1450.0,
      total_pore_volume: 0.78,
      micropore_volume: 0.52,
      carbon: 74.0,
      hydrogen: 2.5,
      nitrogen: 5.4,
      oxygen: 17.6,
      sulfur: 0.5,
      synthesis_recipe: "Slow pyrolysis of crustacean waste at 650°C. High natural pyridinic and pyrrolic N content.",
      advantages: [
        "5.4% Nitrogen Lewis-base sites",
        "Superior low-pressure CO2 affinity (dipole-quadrupole lock)",
        "High selectivity in humid flue streams"
      ]
    },
    {
      id: "coffee-grounds",
      name: "Spent Coffee Grounds Biochar",
      class: "Balanced Hierarchical",
      surface_area: 1180.0,
      total_pore_volume: 0.62,
      micropore_volume: 0.45,
      carbon: 78.5,
      hydrogen: 2.8,
      nitrogen: 2.8,
      oxygen: 15.2,
      sulfur: 0.7,
      synthesis_recipe: "Direct carbonization at 700°C with ZnCl2 pore template. Upcycles urban food service waste.",
      advantages: [
        "Natural 2.8% nitrogen from caffeine residues",
        "Hierarchical meso-micropore transport channels",
        "Zero precursor acquisition cost"
      ]
    },
    {
      id: "microalgae-biochar",
      name: "Spirulina Microalgae Biochar",
      class: "Ultra-High Nitrogen Affinity",
      surface_area: 950.0,
      total_pore_volume: 0.58,
      micropore_volume: 0.39,
      carbon: 68.0,
      hydrogen: 3.2,
      nitrogen: 7.8,
      oxygen: 19.5,
      sulfur: 1.5,
      synthesis_recipe: "Hydrothermal carbonization at 220°C followed by low-temperature activation at 600°C. Preserves organic protein nitrogen.",
      advantages: [
        "Ultra-high 7.8% natural nitrogen content",
        "Exceptional low-concentration / direct air capture affinity",
        "High chemical selectivity over N2 gas"
      ]
    },
    {
      id: "rice-husk",
      name: "Agricultural Rice Husk Biochar",
      class: "Silica-Carbon Composite",
      surface_area: 820.0,
      total_pore_volume: 0.55,
      micropore_volume: 0.38,
      carbon: 65.0,
      hydrogen: 2.1,
      nitrogen: 1.2,
      oxygen: 30.5,
      sulfur: 1.2,
      synthesis_recipe: "Acid leaching of silica followed by NaOH activation at 600°C.",
      advantages: [
        "High thermal and cyclic mechanical stability",
        "Abundant agro-industrial byproduct",
        "Resistant to sintering and pore collapse"
      ]
    },
    {
      id: "pine-sawdust",
      name: "Pine Sawdust Pyrolysis Carbon",
      class: "Low-Cost Baseline",
      surface_area: 540.0,
      total_pore_volume: 0.38,
      micropore_volume: 0.22,
      carbon: 82.0,
      hydrogen: 3.0,
      nitrogen: 0.4,
      oxygen: 14.2,
      sulfur: 0.4,
      synthesis_recipe: "Fast pyrolysis at 550°C under N2 atmosphere without secondary chemical activation.",
      advantages: [
        "Minimal synthesis complexity and chemical usage",
        "Lowest embodied carbon footprint",
        "Economic baseline benchmark"
      ]
    }
  ];

  async function getSelectionGatePrecursors() {
    await wait(100);
    return { count: BIOMASS_PRECURSORS.length, precursors: structuredClone(BIOMASS_PRECURSORS), mock: true };
  }

  async function runSelectionGate({ target_uptake = 3.0, temperature = 25.0, pressure = 1.0, precursor_filter = "all" }) {
    await wait(150);
    const candidates = [];
    for (const item of BIOMASS_PRECURSORS) {
      if (precursor_filter && precursor_filter !== "all" && item.id !== precursor_filter) continue;
      const T_K = temperature + 273.15;
      const sa_factor = item.surface_area / 2000.0;
      const vmicro_factor = item.micropore_volume / 0.6;
      const n_boost = 1.0 + (item.nitrogen * 0.04);
      const press_term = Math.pow(pressure, 0.65) / (0.8 + Math.pow(pressure, 0.65));
      const temp_decay = Math.exp(-0.012 * (T_K - 298.15));
      const pred = Math.max(0.1, Number((5.8 * vmicro_factor * press_term * temp_decay * n_boost).toFixed(4)));
      const delta = Number((pred - target_uptake).toFixed(4));
      const delta_pct = Number(((delta / target_uptake) * 100).toFixed(1));
      const passed = pred >= target_uptake;

      candidates.push({
        id: item.id,
        name: item.name,
        class: item.class,
        gate_status: passed ? "APPROVED" : "REJECTED",
        passed,
        predicted_uptake: pred,
        target_uptake,
        delta,
        delta_pct,
        surface_area: item.surface_area,
        total_pore_volume: item.total_pore_volume,
        micropore_volume: item.micropore_volume,
        carbon: item.carbon,
        hydrogen: item.hydrogen,
        nitrogen: item.nitrogen,
        oxygen: item.oxygen,
        sulfur: item.sulfur,
        nitrogen_pct: item.nitrogen,
        synthesis_recipe: item.synthesis_recipe,
        guidance: passed
          ? `Selection Gate Approved: Meets requirement (+${delta_pct}%). Recommended for synthesis. ${item.synthesis_recipe}`
          : `Selection Gate Filtered Out: Falls short by ${Math.abs(delta_pct)}%. To clear this gate, target pore volume > ${(item.total_pore_volume * 1.3).toFixed(2)} cm³/g or introduce chemical activation with KOH at 750°C.`,
        advantages: item.advantages
      });
    }

    candidates.sort((a, b) => b.predicted_uptake - a.predicted_uptake);
    const passed_count = candidates.filter(c => c.passed).length;
    const top = candidates.find(c => c.passed);

    return {
      status: "success",
      mock: true,
      summary: {
        target_uptake,
        test_temperature: temperature,
        test_pressure: pressure,
        total_screened: candidates.length,
        approved_count: passed_count,
        rejected_count: candidates.length - passed_count,
        top_recommended: top ? top.name : null
      },
      candidates
    };
  }

  return {
    health, getFeatureInfo, getActiveModel, getImportance, getHistory,
    predict, predictBatch,
    getSelectionGatePrecursors, runSelectionGate,
    adminLogin, uploadDataset, startTraining, getTrainingStatus, listModels, activateModel, rollbackModel,
  };
})();

