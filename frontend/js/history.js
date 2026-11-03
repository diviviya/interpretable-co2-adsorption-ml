function historyUser() { try { return JSON.parse(localStorage.getItem("co2_user") || "{}"); } catch (_) { return {}; } }
function escapeHistory(value) { return String(value ?? "—").replace(/[&<>\"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
async function loadPredictionHistory() {
  const status = document.getElementById("history-status"), wrap = document.getElementById("history-table-wrap"), admin = historyUser().role === "admin";
  status.hidden = false; status.textContent = "Loading history…"; wrap.hidden = true;
  try {
    const response = await fetch("/api/history?limit=200", { headers: authHeaders() });
    const data = await response.json().catch(() => ({}));
    if (response.status === 401) { logout(); return; }
    if (!response.ok) throw new Error(data.detail || "Could not load prediction history.");
    const rows = data.history || [];
    if (!rows.length) { status.textContent = admin ? "No user predictions have been recorded yet." : "You have not made any predictions yet."; return; }
    const heads = admin ? ["Date", "Account", "Type", "CO₂ uptake", "Model version"] : ["Date", "Type", "CO₂ uptake", "Model version"];
    const typeLabels = { single: "Single prediction", batch: "Batch prediction", what_if: "What-If analysis" };
    const body = rows.map(row => {
      const date = new Date(row.timestamp); const dateText = Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
      const value = Number(row.predicted_uptake); const uptake = Number.isFinite(value) ? `${value.toFixed(4)} mmol/g` : "—";
      const type = typeLabels[row.prediction_type || "single"] || "Prediction";
      return `<tr><td>${escapeHistory(dateText)}</td>${admin ? `<td>${escapeHistory(row.username || "Unknown")}</td>` : ""}<td>${escapeHistory(type)}</td><td><strong>${uptake}</strong></td><td>${escapeHistory(row.model_version)}</td></tr>`;
    }).join("");
    wrap.innerHTML = `<table class="data-table"><thead><tr>${heads.map(x => `<th>${x}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
    status.hidden = true; wrap.hidden = false;
  } catch (error) { status.textContent = error.message; }
}
document.addEventListener("DOMContentLoaded", () => {
  const admin = historyUser().role === "admin";
  document.getElementById("history-eyebrow").textContent = admin ? "ADMINISTRATION" : "YOUR ACCOUNT";
  document.getElementById("history-title").textContent = admin ? "Prediction history" : "My prediction history";
  document.getElementById("history-lead").textContent = admin ? "Review predictions recorded across all user accounts." : "Review predictions made using your account.";
  document.getElementById("history-section-title").textContent = admin ? "All account predictions" : "Your recent predictions";
  document.getElementById("refresh-history").addEventListener("click", loadPredictionHistory); loadPredictionHistory();
});
