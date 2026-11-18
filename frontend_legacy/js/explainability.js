/**
 * Explainability Dashboard Controller
 */
document.addEventListener("DOMContentLoaded", async () => {
    const globalTableBody = document.getElementById("global-shap-body");

    try {
        const res = await fetch(`${API_BASE}/api/explain`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });
        const data = await res.json();

        if (globalTableBody && data.global_importance) {
            const maxVal = Math.max(...data.global_importance.map(g => Number(g.Mean_Absolute_SHAP)));

            globalTableBody.innerHTML = data.global_importance.map((g, idx) => {
                const val = Number(g.Mean_Absolute_SHAP);
                const pct = Math.round((val / maxVal) * 100);
                return `
                    <tr>
                        <td><b>#${idx + 1}</b></td>
                        <td><b>${g.Feature}</b></td>
                        <td>${val.toFixed(4)}</td>
                        <td>
                            <div style="background: #edf2ed; height: 10px; border-radius: 5px; overflow: hidden; width: 140px;">
                                <div style="background: var(--green); height: 100%; width: ${pct}%;"></div>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }
    } catch (e) {
        console.error("Failed to load explainability data:", e);
    }
});
