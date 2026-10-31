/**
 * History Controller
 */

let historyRecords = [];

const historyTbody = document.getElementById("history-tbody");
const btnRefresh = document.getElementById("btn-refresh-history");
const btnExport = document.getElementById("btn-export-history");

async function loadHistory() {
    historyTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 30px; color: var(--muted);">Loading history records...</td></tr>`;

    try {
        const res = await fetch("/api/history");
        const data = await res.json();
        historyRecords = data.predictions || [];

        if (historyRecords.length === 0) {
            historyTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 40px; color: var(--muted);">No prediction history recorded yet. Make a single or batch prediction to see records here.</td></tr>`;
            return;
        }

        historyTbody.innerHTML = "";
        historyRecords.slice().reverse().forEach(rec => {
            const tr = document.createElement("tr");
            const dt = rec.timestamp ? new Date(rec.timestamp).toLocaleString() : "-";
            const inputs = rec.inputs || {};

            tr.innerHTML = `
                <td><small style="color: var(--muted);">${dt}</small></td>
                <td><b>${rec.username || "anonymous"}</b></td>
                <td><span class="badge-tag" style="font-size: 11px;">${rec.model_version || "v1.0"}</span></td>
                <td>${inputs.surface_area ?? "-"}</td>
                <td>${inputs.temperature ?? "-"}</td>
                <td>${inputs.pressure ?? "-"}</td>
                <td><b style="color: var(--green);">${rec.predicted_uptake !== undefined ? Number(rec.predicted_uptake).toFixed(3) : "-"}</b></td>
            `;
            historyTbody.appendChild(tr);
        });
    } catch (err) {
        historyTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 30px; color: var(--danger);">Failed to load history: ${err.message}</td></tr>`;
    }
}

btnRefresh.addEventListener("click", loadHistory);

btnExport.addEventListener("click", () => {
    if (historyRecords.length === 0) return;
    const headers = ["Timestamp", "User", "Model Version", "Surface Area (m2/g)", "Temp (C)", "Pressure (bar)", "Predicted CO2 (mmol/g)"];
    const rows = [headers.join(",")];

    historyRecords.forEach(r => {
        const row = [
            `"${r.timestamp || ""}"`,
            `"${r.username || "anonymous"}"`,
            `"${r.model_version || ""}"`,
            r.inputs?.surface_area ?? "",
            r.inputs?.temperature ?? "",
            r.inputs?.pressure ?? "",
            r.predicted_uptake ?? ""
        ];
        rows.push(row.join(","));
    });

    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `prediction_history_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
});

document.addEventListener("DOMContentLoaded", loadHistory);
