const token = localStorage.getItem("co2_access_token");
const authStatus = document.getElementById("auth-status");
const datasetSelect = document.getElementById("dataset-select");
const uploadButton = document.getElementById("upload-btn");
const trainButton = document.getElementById("train-btn");
let uploadedDataset = null;
const adminHeaders = () => ({ Authorization: `Bearer ${token}` });
async function adminApi(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...adminHeaders(), ...(options.headers || {}) } });
  const data = await response.json().catch(() => ({}));
  if (response.status === 401 || response.status === 403) throw new Error(data.detail || "Administrator access is required.");
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
async function initialiseAdmin() {
  try { const me = await adminApi("/api/auth/me"); if (me.role !== "admin") throw new Error("This page is available to administrators only."); authStatus.textContent = `Administrator access confirmed · ${me.username}`; await loadModels(); }
  catch (error) { authStatus.textContent = error.message; uploadButton.disabled = true; trainButton.disabled = true; }
}
uploadButton.addEventListener("click", async () => {
  const file = document.getElementById("dataset-file").files[0]; const status = document.getElementById("upload-status");
  if (!file) { status.textContent = "Choose a CSV file first."; return; }
  uploadButton.disabled = true; status.textContent = "Uploading and validating dataset…";
  try {
    const form = new FormData(); form.append("file", file);
    const data = await adminApi("/api/admin/datasets", { method: "POST", body: form });
    uploadedDataset = data.report.stored_filename; datasetSelect.innerHTML = "";
    const option = document.createElement("option"); option.value = uploadedDataset; option.textContent = `${data.report.filename} · ${data.report.row_count} rows`; datasetSelect.append(option);
    trainButton.disabled = false; status.textContent = `Dataset validated: ${data.report.row_count} rows.`;
  } catch (error) { status.textContent = error.message; }
  finally { uploadButton.disabled = false; }
});
trainButton.addEventListener("click", async () => {
  const status = document.getElementById("training-status"); const datasetName = datasetSelect.value || uploadedDataset;
  if (!datasetName) { status.textContent = "Upload and validate a dataset first."; return; }
  trainButton.disabled = true; status.textContent = "Starting training…";
  try {
    const started = await adminApi("/api/admin/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_name: datasetName }) });
    const jobId = started.job.job_id; let complete = false;
    while (!complete) {
      await new Promise(resolve => setTimeout(resolve, 1800));
      const job = await adminApi(`/api/admin/training/${encodeURIComponent(jobId)}`);
      status.textContent = `${String(job.status).toUpperCase()} · ${job.progress_pct ?? 0}% · ${job.stage || ""}`;
      if (job.status === "completed") { status.textContent = `Training completed. Candidate version: ${job.candidate_version}. Review it below before activation. Validation metrics: ${JSON.stringify(job.metrics || {})}`; complete = true; await loadModels(); }
      if (job.status === "failed") throw new Error(job.error || "Training failed.");
    }
  } catch (error) { status.textContent = error.message; }
  finally { trainButton.disabled = false; }
});
async function loadModels() {
  const container = document.getElementById("models-list"); container.textContent = "Loading model versions…";
  try {
    const data = await adminApi("/api/admin/models"); container.innerHTML = "";
    if (!data.models?.length) { container.textContent = "No registered model versions."; return; }
    data.models.forEach(model => {
      const row = document.createElement("article"); row.className = "model-row";
      const details = document.createElement("div"); details.className = "model-row-details";
      const title = document.createElement("h3"); title.textContent = model.name || model.version;
      const meta = document.createElement("p"); meta.textContent = `${model.version} · ${model.status}${model.metrics ? ` · Validation R² ${model.metrics.R2 ?? "—"} · RMSE ${model.metrics.RMSE ?? "—"}` : ""}`;
      details.append(title, meta); const actions = document.createElement("div"); actions.className = "model-row-actions";
      const badge = document.createElement("span"); badge.className = `status-pill ${model.status === "active" ? "is-active" : ""}`; badge.textContent = model.status; actions.append(badge);
      if (model.status !== "active" && model.artifact_path) { const activate = document.createElement("button"); activate.className = "button-secondary"; activate.textContent = "Activate"; activate.addEventListener("click", async () => { if (!confirm(`Activate ${model.version}? Future predictions will use this version.`)) return; try { await adminApi(`/api/admin/models/${encodeURIComponent(model.version)}/activate`, { method: "POST" }); await loadModels(); } catch (error) { alert(error.message); } }); actions.append(activate); }
      row.append(details, actions); container.append(row);
    });
  } catch (error) { container.textContent = error.message; }
}
document.getElementById("refresh-models")?.addEventListener("click", loadModels);
initialiseAdmin();
