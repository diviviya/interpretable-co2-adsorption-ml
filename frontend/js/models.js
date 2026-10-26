/**
 * Model Comparison Dashboard Controller
 */
document.addEventListener("DOMContentLoaded", async () => {
    const tableBody = document.getElementById("models-table-body");
    const weightsBody = document.getElementById("weights-table-body");
    const r2Chart = document.getElementById("r2-bars");

    try {
        const res = await fetch(`${API_BASE}/api/models`);
        const data = await res.json();

        // Render Evaluation Metrics Table
        if (tableBody && data.evaluation_metrics) {
            tableBody.innerHTML = data.evaluation_metrics.map(m => {
                const isBest = m.Model.includes("Ensemble");
                return `
                    <tr class="${isBest ? 'highlight-row' : ''}">
                        <td>
                            <b>${m.Model}</b>
                            ${isBest ? '<span class="badge-tag badge-best" style="margin-left: 6px;">Selected</span>' : ''}
                        </td>
                        <td>${Number(m.MAE).toFixed(4)}</td>
                        <td>${Number(m.RMSE).toFixed(4)}</td>
                        <td><b>${Number(m.R2).toFixed(4)}</b></td>
                    </tr>
                `;
            }).join('');
        }

        // Render Ensemble Weights Table
        if (weightsBody && data.ensemble_weights) {
            weightsBody.innerHTML = data.ensemble_weights.map(w => {
                const optWeight = Number(w.Validation_Optimized_Weight || 0);
                const pct = (optWeight * 100).toFixed(1);
                return `
                    <tr>
                        <td><b>${w.Model}</b></td>
                        <td>${optWeight.toFixed(4)} (${pct}%)</td>
                        <td>
                            <div style="background: #eef2ed; height: 8px; border-radius: 4px; overflow: hidden; width: 120px;">
                                <div style="background: var(--green); height: 100%; width: ${pct}%;"></div>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        // Render R2 Comparison Bars
        if (r2Chart && data.evaluation_metrics) {
            r2Chart.innerHTML = data.evaluation_metrics.map(m => {
                const r2Val = Number(m.R2);
                const pct = Math.max(10, Math.round(r2Val * 100));
                const isBest = m.Model.includes("Ensemble");
                return `
                    <div style="margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 650; margin-bottom: 4px;">
                            <span>${m.Model}</span>
                            <span>R² = ${r2Val.toFixed(4)}</span>
                        </div>
                        <div style="height: 12px; background: #edf2ed; border-radius: 6px; overflow: hidden;">
                            <div style="height: 100%; width: ${pct}%; background: ${isBest ? 'var(--green)' : '#8aa295'}; border-radius: 6px;"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    } catch (e) {
        console.error("Failed to load model evaluation metrics:", e);
    }
});
