// =====================================================================
// batch.js — Page C: upload a CSV, predict every row, download results.
// =====================================================================

renderNav("batch");

const PRED_COL = "Predicted CO2 uptake (mmol/g)";
let chosenFile = null;
let lastResult = null;

$("required-cols").innerHTML = FEATURE_COLUMNS.map((c) => `<code>${escapeHtml(c)}</code>`).join(", ") +
  `. A <code>${escapeHtml(TARGET)}</code> column is optional and ignored. Optional inputs can be left empty.`;

function setFileAlerts(items) {
  $("file-alerts").innerHTML = items.map(([kind, t]) => `<li class="${kind}">${escapeHtml(t)}</li>`).join("");
}

function tableHtml(columns, rows, highlight) {
  const head = columns.map((c) => `<th class="${c === highlight ? "num" : ""}">${escapeHtml(c)}</th>`).join("");
  const body = rows.map((r) =>
    "<tr>" + columns.map((c) => `<td class="${c === highlight ? "num highlight" : ""}">${escapeHtml(r[c])}</td>`).join("") + "</tr>"
  ).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// ---------- Step 1: pick a file ----------
async function handleFile(file) {
  chosenFile = null;
  $("preview-card").hidden = true;
  $("results-card").hidden = true;
  $("file-name").textContent = file ? file.name : "";

  const problem = checkCsvFile(file);
  if (problem) { setFileAlerts([["error", problem]]); return; }

  let parsed;
  try { parsed = parseCSV(await file.text()); }
  catch { setFileAlerts([["error", "Couldn't read this file as text."]]); return; }

  if (!parsed.rows.length) { setFileAlerts([["error", "The file has a header but no data rows."]]); return; }
  const missing = FEATURE_COLUMNS.filter((c) => !parsed.columns.includes(c));
  if (missing.length) {
    setFileAlerts([["error", "These required columns are missing (names must match exactly): " + missing.join(", ")]]);
    return;
  }

  chosenFile = file;
  setFileAlerts([["ok", `${parsed.rows.length} rows found with all 10 required columns.`]]);

  // ---------- Step 2: preview ----------
  const shown = parsed.rows.slice(0, 10);
  $("preview-sub").textContent = `Showing the first ${shown.length} of ${parsed.rows.length} rows. The server will check every value before predicting.`;
  $("preview-table").innerHTML = tableHtml(FEATURE_COLUMNS, shown);
  $("preview-card").hidden = false;
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

// ---------- Step 3: run ----------
$("run-btn").addEventListener("click", async () => {
  if (!chosenFile) return;
  const btn = $("run-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';
  try {
    lastResult = await API.predictBatch(chosenFile);
    showResults(lastResult);
  } catch (err) {
    $("results-card").hidden = true;
    setFileAlerts([["error", err.message]]);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run predictions";
  }
});

function showResults(res) {
  const rows = res.rows.map((r) => ({ ...r, [PRED_COL]: fmt(r.prediction) }));
  const errors = res.errors || [];
  $("results-card").hidden = false;
  $("mock-badge").hidden = !res.mock;
  $("results-sub").textContent =
    `${rows.length} rows predicted with model ${res.model_version}` + (errors.length ? `, ${errors.length} row${errors.length === 1 ? "" : "s"} skipped.` : ".");
  $("row-errors").innerHTML = errors.slice(0, 20)
    .map((e) => `<li class="error">Row ${escapeHtml(e.row)}: ${escapeHtml(e.message)}</li>`).join("") +
    (errors.length > 20 ? `<li class="error">…and ${errors.length - 20} more rows with problems.</li>` : "");
  $("results-table").innerHTML = tableHtml(["row", ...FEATURE_COLUMNS, PRED_COL], rows, PRED_COL);
  $("download-btn").disabled = !rows.length;
}

$("download-btn").addEventListener("click", () => {
  if (!lastResult) return;
  const rows = lastResult.rows.map((r) => ({ ...r, [PRED_COL]: r.prediction, model_version: lastResult.model_version }));
  const columns = ["row", ...FEATURE_COLUMNS, PRED_COL, "model_version"];
  downloadText("co2_predictions.csv", toCSV(columns, rows));
});

$("template-btn").addEventListener("click", () => {
  downloadText("co2_batch_template.csv", FEATURE_COLUMNS.map((c) => (/[",]/.test(c) ? `"${c}"` : c)).join(",") + "\n");
});
