/**
 * CO2 Uptake Prediction & What-If Scenario Controller
 */

// =============================================================================
// 1. TAB NAVIGATION CONTROLLER
// =============================================================================
const tabBtnSingle = document.getElementById("tab-btn-single");
const tabBtnWhatif = document.getElementById("tab-btn-whatif");
const paneSingle = document.getElementById("pane-single");
const paneWhatif = document.getElementById("pane-whatif");

function switchTab(target) {
    if (target === "whatif") {
        tabBtnSingle.classList.remove("active");
        tabBtnWhatif.classList.add("active");
        paneSingle.classList.remove("active");
        paneWhatif.classList.add("active");
        window.location.hash = "what-if";
    } else {
        tabBtnWhatif.classList.remove("active");
        tabBtnSingle.classList.add("active");
        paneWhatif.classList.remove("active");
        paneSingle.classList.add("active");
        window.location.hash = "";
    }
}

if (tabBtnSingle) tabBtnSingle.addEventListener("click", () => switchTab("single"));
if (tabBtnWhatif) tabBtnWhatif.addEventListener("click", () => switchTab("whatif"));

// Check URL hash on load
if (window.location.hash === "#what-if") {
    switchTab("whatif");
}

// =============================================================================
// 2. SINGLE PREDICTION CONTROLLER
// =============================================================================
const form = document.getElementById("prediction-form");
const resultBox = document.getElementById("prediction-result");
const predictBtn = document.getElementById("btn-predict");
const exampleBtn = document.getElementById("btn-example");
const clearBtn = document.getElementById("btn-clear");
const quickWhatIfCta = document.getElementById("quick-whatif-cta");
const btnSendToWhatIf = document.getElementById("btn-send-to-whatif");

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
        if (quickWhatIfCta) quickWhatIfCta.style.display = "none";
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
            if (quickWhatIfCta) quickWhatIfCta.style.display = "block";
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

// Transfer from prediction form to What-If baseline
if (btnSendToWhatIf) {
    btnSendToWhatIf.addEventListener("click", () => {
        const rawData = Object.fromEntries(new FormData(form));
        copyToWhatIfBaseline(rawData);
        switchTab("whatif");
    });
}

// =============================================================================
// 3. WHAT-IF SCENARIO ANALYSIS CONTROLLER
// =============================================================================
const whatIfForm = document.getElementById("whatif-form");
const whatIfResult = document.getElementById("whatif-result");
const whatIfBtn = document.getElementById("btn-whatif");
const presetHighPBtn = document.getElementById("btn-preset-highp");
const btnCopyFromPred = document.getElementById("btn-copy-from-pred");
const whatifDetailsPanel = document.getElementById("whatif-details-panel");

const whatIfDefaultBaseline = {
    surface_area: 1200,
    total_pore_volume: 0.65,
    micropore_volume: 0.40,
    carbon_pct: 75.0,
    hydrogen_pct: 2.0,
    nitrogen_pct: 1.5,
    oxygen_pct: 20.0,
    sulfur_pct: 0.2,
    temperature_c: 25,
    pressure_bar: 1.0
};

const whatIfKeys = Object.keys(whatIfDefaultBaseline);

function populateWhatIf(base, mod) {
    whatIfKeys.forEach(k => {
        const baseEl = document.getElementById(`base_${k}`);
        const modEl = document.getElementById(`mod_${k}`);
        if (baseEl && base[k] !== undefined) baseEl.value = base[k];
        if (modEl && mod[k] !== undefined) modEl.value = mod[k];
    });
    highlightWhatIfChanges();
}

function copyToWhatIfBaseline(data) {
    whatIfKeys.forEach(k => {
        const baseEl = document.getElementById(`base_${k}`);
        const modEl = document.getElementById(`mod_${k}`);
        const val = data[k] !== undefined && data[k] !== "" ? data[k] : (whatIfDefaultBaseline[k] || 0);
        if (baseEl) baseEl.value = val;
        if (modEl) modEl.value = val;
    });
    highlightWhatIfChanges();
}

function highlightWhatIfChanges() {
    whatIfKeys.forEach(k => {
        const baseEl = document.getElementById(`base_${k}`);
        const modEl = document.getElementById(`mod_${k}`);
        if (baseEl && modEl) {
            const b = Number(baseEl.value);
            const m = Number(modEl.value);
            if (!isNaN(b) && !isNaN(m) && b !== m) {
                modEl.style.borderColor = "var(--green)";
                modEl.style.background = "#eef8f2";
            } else {
                modEl.style.borderColor = "var(--line)";
                modEl.style.background = "#fbfdfb";
            }
        }
    });
}

// Initial populate for What-If
populateWhatIf(whatIfDefaultBaseline, {
    ...whatIfDefaultBaseline,
    pressure_bar: 2.5,
    temperature_c: 0.0
});

// Highlight modified inputs when user types
whatIfKeys.forEach(k => {
    const modEl = document.getElementById(`mod_${k}`);
    if (modEl) {
        modEl.addEventListener("input", highlightWhatIfChanges);
    }
});

if (presetHighPBtn) {
    presetHighPBtn.addEventListener("click", () => {
        populateWhatIf(whatIfDefaultBaseline, {
            ...whatIfDefaultBaseline,
            surface_area: 1650,
            nitrogen_pct: 4.5,
            pressure_bar: 2.5
        });
    });
}

if (btnCopyFromPred) {
    btnCopyFromPred.addEventListener("click", () => {
        const rawData = Object.fromEntries(new FormData(form));
        copyToWhatIfBaseline(rawData);
    });
}

if (whatIfForm) {
    whatIfForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const basePayload = {};
        const modPayload = {};

        whatIfKeys.forEach(k => {
            const bVal = Number(document.getElementById(`base_${k}`).value);
            const mVal = Number(document.getElementById(`mod_${k}`).value);
            basePayload[k] = isNaN(bVal) ? 0 : bVal;
            modPayload[k] = isNaN(mVal) ? 0 : mVal;
        });

        whatIfBtn.disabled = true;
        whatIfBtn.textContent = "Simulating Scenario...";

        try {
            const res = await fetch(`${API_BASE}/api/what-if`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    baseline: basePayload,
                    modified: modPayload
                })
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Simulation failed.");

            renderWhatIfResults(data);
        } catch (err) {
            whatIfResult.innerHTML = `
                <div style="padding: 20px; color: var(--danger); text-align: center;">
                    <h3>What-If Simulation Error</h3>
                    <p>${err.message}</p>
                </div>
            `;
        } finally {
            whatIfBtn.disabled = false;
            whatIfBtn.textContent = "Run What-If Comparison →";
        }
    });
}

function renderWhatIfResults(data) {
    const isGain = data.delta_mmol_g >= 0;
    const deltaFormatted = `${isGain ? '+' : ''}${data.delta_mmol_g.toFixed(3)}`;
    const pctFormatted = `${isGain ? '+' : ''}${data.percentage_change.toFixed(1)}%`;

    whatIfResult.innerHTML = `
        <div class="what-if-summary">
            <div class="what-if-box">
                <small>Baseline Prediction</small>
                <strong>${data.baseline_prediction.toFixed(3)}</strong>
                <span style="font-size: 13px; color: var(--muted); margin-left: 4px;">mmol/g</span>
            </div>

            <div style="font-size: 24px; color: var(--muted);">➔</div>

            <div class="what-if-box">
                <small>Modified Prediction</small>
                <strong>${data.modified_prediction.toFixed(3)}</strong>
                <span style="font-size: 13px; color: var(--muted); margin-left: 4px;">mmol/g</span>
            </div>

            <div class="what-if-box">
                <small>Difference (Delta)</small>
                <div class="what-if-delta ${isGain ? 'delta-pos' : 'delta-neg'}">
                    ${deltaFormatted} mmol/g (${pctFormatted})
                </div>
            </div>
        </div>
    `;

    if (whatifDetailsPanel) {
        whatifDetailsPanel.innerHTML = `
            <div class="panel" style="margin-top: 10px;">
                <h3 style="font-size: 16px; font-weight: 750; margin-bottom: 14px;">Changed Parameters Breakdown</h3>
                ${data.changed_features.length === 0 ? `
                    <p style="font-size: 13.5px; color: var(--muted);">No input parameters were changed between baseline and modified.</p>
                ` : `
                    <div class="table-wrapper">
                        <table>
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Baseline Value</th>
                                    <th>Modified Value</th>
                                    <th>Difference</th>
                                    <th>% Change</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.changed_features.map(f => {
                                    const diffSign = f.difference >= 0 ? '+' : '';
                                    return `
                                        <tr>
                                            <td><b>${f.feature}</b></td>
                                            <td>${f.baseline_value}</td>
                                            <td><b style="color: var(--green);">${f.modified_value}</b></td>
                                            <td>${diffSign}${f.difference}</td>
                                            <td><b>${diffSign}${f.pct_change}%</b></td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                `}
            </div>
        `;
    }
}
