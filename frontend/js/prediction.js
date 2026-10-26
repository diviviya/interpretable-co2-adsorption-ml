/**
 * CO2 Uptake Prediction Controller
 */
const form = document.getElementById("prediction-form");
const resultBox = document.getElementById("prediction-result");
const predictBtn = document.getElementById("btn-predict");
const exampleBtn = document.getElementById("btn-example");
const clearBtn = document.getElementById("btn-clear");

const standardExample = {
    surface_area: 1420,
    total_pore_volume: 0.74,
    micropore_volume: 0.48,
    carbon_pct: 78.855,
    hydrogen_pct: 1.5,
    nitrogen_pct: 2.26,
    oxygen_pct: 15.05,
    sulfur_pct: 0,
    temperature_c: 25,
    pressure_bar: 1
};

if (exampleBtn) {
    exampleBtn.addEventListener("click", () => {
        Object.entries(standardExample).forEach(([key, val]) => {
            if (form.elements[key]) form.elements[key].value = val;
        });
    });
}

if (clearBtn) {
    clearBtn.addEventListener("click", () => {
        form.reset();
        resultBox.innerHTML = `
            <div style="text-align: center; color: var(--muted); padding: 40px 20px;">
                <div style="font-size: 32px; margin-bottom: 10px;">↗</div>
                <h3 style="font-size: 16px; font-weight: 700; margin-bottom: 6px;">Your prediction will appear here</h3>
                <p style="font-size: 13px;">Fill in the material properties and adsorption conditions, then click Predict.</p>
            </div>
        `;
    });
}

if (form) {
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const rawData = Object.fromEntries(new FormData(form));
        const payload = {};
        for (const [k, v] of Object.entries(rawData)) {
            const num = Number(v);
            if (isNaN(num)) {
                alert(`Please enter a valid numeric value for ${k}`);
                return;
            }
            payload[k] = num;
        }

        predictBtn.disabled = true;
        predictBtn.textContent = "Calculating Prediction...";

        try {
            const response = await fetch(`${API_BASE}/api/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "Prediction request failed.");
            }

            renderPrediction(data);
        } catch (err) {
            resultBox.innerHTML = `
                <div style="padding: 24px; text-align: center; color: var(--danger);">
                    <h3>Prediction Error</h3>
                    <p style="font-size: 13px; margin-top: 6px;">${err.message}</p>
                </div>
            `;
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict CO₂ Uptake →";
        }
    });
}

function renderPrediction(data) {
    const val = Number(data.predicted_co2_uptake_mmol_g).toFixed(3);
    const shap = data.shap_contributions || [];
    const maxShap = Math.max(...shap.map(s => Math.abs(s.shap_value)), 0.001);

    resultBox.innerHTML = `
        <div style="width: 100%;">
            <div class="prediction-card">
                <div class="prediction-val">${val}</div>
                <div class="prediction-unit">mmol/g &bull; Predicted CO₂ Uptake</div>
            </div>

            <div class="meta-row">
                <span>Model Strategy</span>
                <b>${data.model || "Weighted Ensemble"}</b>
            </div>

            ${shap.length > 0 ? `
                <div class="shap-container">
                    <div class="shap-header">
                        <span style="font-weight: 750; font-size: 13px;">SHAP Feature Attribution</span>
                        <span style="font-size: 11px; color: var(--muted);">Base: ${data.base_value || '3.49'} mmol/g</span>
                    </div>
                    <p style="font-size: 11.5px; color: var(--muted); margin-bottom: 12px;">
                        Exact marginal contributions driving the prediction above or below dataset average:
                    </p>
                    <div class="shap-list">
                        ${shap.slice(0, 5).map(item => {
                            const isPos = item.shap_value >= 0;
                            const pct = Math.min(100, Math.max(10, Math.round((Math.abs(item.shap_value) / maxShap) * 100)));
                            return `
                                <div class="shap-item">
                                    <div class="shap-labels">
                                        <span>${item.feature}: <b>${item.input_value}</b></span>
                                        <span class="shap-val-badge ${isPos ? 'val-pos' : 'val-neg'}">${item.formatted_shap} mmol/g</span>
                                    </div>
                                    <div class="shap-track">
                                        <div class="shap-fill ${isPos ? 'fill-pos' : 'fill-neg'}" style="width: ${pct}%;"></div>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}
