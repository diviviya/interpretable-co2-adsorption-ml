// =====================================================================
// history.js — Page D: previous predictions, fetched from the backend.
// =====================================================================

renderNav("history");

// Columns shown in the table: time, version, prediction, then the inputs.
const SHOWN = ["Surface Area (m2/g)", "N (%)", "Temp (°C)", "Pressure (bar)"];

async function loadHistory() {
  $("history-alerts").innerHTML = "";
  $("history-sub").textContent = "Loading…";
  try {
    const res = await API.getHistory();
    const items = res.items || [];
    $("history-sub").textContent = items.length
      ? `${items.length} prediction${items.length === 1 ? "" : "s"}${res.mock ? " (demo data, this browser tab only)" : ""}`
      : "";
    $("history-empty").hidden = items.length > 0;
    $("history-table").hidden = items.length === 0;
    if (!items.length) return;

    const head = ["Time", "Model version", "Prediction (mmol/g)", ...SHOWN.map((c) => {
      const f = FIELDS.find((x) => x.column === c);
      return `${f.label} (${f.unit})`;
    })];
    const body = items.map((it) => `
      <tr>
        <td>${escapeHtml(fmtDate(it.timestamp))}</td>
        <td>${escapeHtml(it.model_version)}</td>
        <td class="num highlight">${escapeHtml(fmt(it.prediction))}</td>
        ${SHOWN.map((c) => `<td class="num">${it.features && it.features[c] != null ? escapeHtml(fmt(it.features[c])) : "–"}</td>`).join("")}
      </tr>`).join("");
    $("history-table").innerHTML =
      `<table><thead><tr>${head.map((h, i) => `<th class="${i >= 2 ? "num" : ""}">${escapeHtml(h)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`;
  } catch (err) {
    $("history-sub").textContent = "";
    $("history-table").hidden = true;
    $("history-empty").hidden = true;
    $("history-alerts").innerHTML = `<li class="error">${escapeHtml(err.message)}</li>`;
  }
}

$("refresh-btn").addEventListener("click", loadHistory);
loadHistory();
