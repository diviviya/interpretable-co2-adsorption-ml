const batchForm = document.getElementById("batch-prediction-form");
const batchFileInput = document.getElementById("batch-csv-file");
const batchButton = document.getElementById("btn-batch-predict");
const batchStatus = document.getElementById("batch-prediction-status");
const batchResult = document.getElementById("batch-prediction-result");
let lastBatchFile = null;
function batchEscape(value) { return String(value ?? "—").replace(/[&<>\"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
if (batchForm) batchForm.addEventListener("submit", async event => {
  event.preventDefault(); const file = batchFileInput.files[0];
  if (!file) { batchStatus.textContent = "Choose a CSV file first."; return; }
  if (!file.name.toLowerCase().endsWith(".csv")) { batchStatus.textContent = "Only CSV files are supported."; return; }
  if (file.size > 15 * 1024 * 1024) { batchStatus.textContent = "The CSV must be 15 MB or smaller."; return; }
  lastBatchFile = file; batchButton.disabled = true; batchStatus.textContent = "Processing records…"; batchResult.innerHTML = "";
  try {
    const form = new FormData(); form.append("file", file);
    const response = await fetch("/api/predict/batch", { method: "POST", headers: authHeaders(), body: form });
    const data = await response.json().catch(() => ({}));
    if (response.status === 401) { logout(); return; }
    if (!response.ok) throw new Error(data.detail || "Batch prediction failed.");
    const preview = (data.predictions_preview || []).map(row => {
      const values = Object.entries(row).filter(([key]) => key !== "predicted_co2_uptake_mmol_g").slice(0, 3).map(([, value]) => `<td>${batchEscape(value)}</td>`).join("");
      const uptake = Number(row.predicted_co2_uptake_mmol_g);
      return `<tr>${values}<td><strong>${Number.isFinite(uptake) ? uptake.toFixed(4) : "—"}</strong></td></tr>`;
    }).join("");
    batchStatus.textContent = `${data.rows_processed} records processed successfully.`;
    batchResult.innerHTML = `<div class="batch-summary"><div><span>Mean predicted uptake</span><strong>${Number(data.mean_uptake_mmol_g).toFixed(4)} mmol/g</strong></div><div><span>Predicted range</span><strong>${Number(data.min_uptake_mmol_g).toFixed(4)} – ${Number(data.max_uptake_mmol_g).toFixed(4)} mmol/g</strong></div></div><h2>Result preview</h2><div class="table-scroll"><table class="data-table"><thead><tr><th>Input 1</th><th>Input 2</th><th>Input 3</th><th>Predicted uptake (mmol/g)</th></tr></thead><tbody>${preview}</tbody></table></div><button id="download-batch" class="button-primary" type="button">Download complete results CSV ↓</button>`;
    document.getElementById("download-batch").addEventListener("click", downloadBatch);
  } catch (error) { batchStatus.textContent = error.message; }
  finally { batchButton.disabled = false; }
});
async function downloadBatch() {
  if (!lastBatchFile) return; const button = document.getElementById("download-batch"); button.disabled = true; button.textContent = "Preparing download…";
  try {
    const form = new FormData(); form.append("file", lastBatchFile);
    const response = await fetch("/api/predict/batch/download", { method: "POST", headers: authHeaders(), body: form });
    if (response.status === 401) { logout(); return; }
    if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.detail || "Download failed."); }
    const blob = await response.blob(), url = URL.createObjectURL(blob), anchor = document.createElement("a");
    anchor.href = url; anchor.download = `predicted_${lastBatchFile.name}`; document.body.append(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
  } catch (error) { batchStatus.textContent = error.message; }
  finally { button.disabled = false; button.textContent = "Download complete results CSV ↓"; }
}
