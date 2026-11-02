function getHistoryUser() {
  try {
    return JSON.parse(localStorage.getItem("co2_user") || "{}");
  } catch (_) {
    return {};
  }
}

function escapeHistoryText(value) {
  return String(value ?? "—").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}

async function loadPredictionHistory() {
  const status = document.getElementById("history-status");
  const wrap = document.getElementById("history-table-wrap");
  const user = getHistoryUser();
  const isAdmin = user.role === "admin";

  status.hidden = false;
  status.textContent = isAdmin ? "Loading prediction history…" : "Loading your history…";
  wrap.hidden = true;

  try {
    const response = await fetch(`${API_BASE}/api/history`, { headers: authHeaders() });
    const data = await response.json();
    if (response.status === 401) { logout(); return; }
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : "Could not load prediction history.";
      throw new Error(detail);
    }

    const rows = data.history || [];
    if (!rows.length) {
      status.textContent = isAdmin ? "No predictions have been recorded yet." : "You haven't made any predictions yet.";
      return;
    }

    const headers = isAdmin
      ? "<th>Date</th><th>Account</th><th>CO₂ uptake</th><th>Model version</th>"
      : "<th>Date</th><th>CO₂ uptake</th><th>Model version</th>";

    const body = rows.map(row => {
      const date = new Date(row.timestamp);
      const dateText = Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
      const accountCell = isAdmin ? `<td>${escapeHistoryText(row.username || "Unknown account")}</td>` : "";
      return `<tr>
        <td>${escapeHistoryText(dateText)}</td>
        ${accountCell}
        <td><strong>${Number(row.predicted_uptake).toFixed(4)} mmol/g</strong></td>
        <td>${escapeHistoryText(row.model_version || "—")}</td>
      </tr>`;
    }).join("");

    wrap.innerHTML = `<table class="history-table"><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table>`;
    status.hidden = true;
    wrap.hidden = false;
  } catch (error) {
    status.textContent = error.message || "Could not load history.";
    status.hidden = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (!getAccessToken()) {
    window.location.replace("login.html?next=history.html");
    return;
  }

  const isAdmin = getHistoryUser().role === "admin";
  document.querySelector(".eyebrow").textContent = isAdmin ? "ADMINISTRATION" : "YOUR ACCOUNT";
  document.querySelector(".lead").textContent = isAdmin
    ? "Review predictions made across user accounts."
    : "A record of predictions made using your account.";
  document.querySelector(".history-card h2").textContent = isAdmin ? "Recent predictions across accounts" : "Recent predictions";
  document.getElementById("refresh-history").addEventListener("click", loadPredictionHistory);
  loadPredictionHistory();
});
