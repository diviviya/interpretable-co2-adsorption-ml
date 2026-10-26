/**
 * What-If Scenario Analysis Controller
 */
const whatIfForm = document.getElementById("whatif-form");
const whatIfResult = document.getElementById("whatif-result");
const whatIfBtn = document.getElementById("btn-whatif");
const presetBtn = document.getElementById("btn-preset");

const baselineDefaults = {
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

function populateBaselineAndModified(base, mod) {
    Object.entries(base).forEach(([k, v]) => {
        const el = document.getElementById(`base_${k}`);
        if (el) el.value = v;
    });

    Object.entries(mod).forEach(([k, v]) => {
        const el = document.getElementById(`mod_${k}`);
        if (el) el.value = v;
    });
    highlightChanges();
}

function highlightChanges() {
    const keys = Object.keys(baselineDefaults);
    keys.forEach(k => {
        const baseEl = document.getElementById(`base_${k}`);
        const modEl = document.getElementById(`mod_${k}`);
        if (baseEl && modEl) {
            if (Number(baseEl.value) !== Number(modEl.value)) {
                modEl.style.borderColor = "var(--green)";
                modEl.style.background = "#eef8f2";
            } else {
                modEl.style.borderColor = "var(--line)";
                modEl.style.background = "#fbfdfb";
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    // Populate defaults on initial load
    populateBaselineAndModified(baselineDefaults, {
        ...baselineDefaults,
        pressure_bar: 2.5,
        temperature_c: 0.0
    });

    // Listen to changes to highlight modifications
    const keys = Object.keys(baselineDefaults);
    keys.forEach(k => {
        const modEl = document.getElementById(`mod_${k}`);
        if (modEl) {
            modEl.addEventListener("input", highlightChanges);
        }
    });
});

if (presetBtn) {
    presetBtn.addEventListener("click", () => {
        populateBaselineAndModified(baselineDefaults, {
            ...baselineDefaults,
            surface_area: 1650,
            nitrogen_pct: 4.5,
            pressure_bar: 2.0
        });
    });
}

if (whatIfForm) {
    whatIfForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const basePayload = {};
        const modPayload = {};
        const keys = Object.keys(baselineDefaults);

        keys.forEach(k => {
            const bVal = Number(document.getElementById(`base_${k}`).value);
            const mVal = Number(document.getElementById(`mod_${k}`).value);
            basePayload[k] = isNaN(bVal) ? 0 : bVal;
            modPayload[k] = isNaN(mVal) ? 0 : mVal;
        });

        whatIfBtn.disabled = true;
        whatIfBtn.textContent = "Simulating What-If...";

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
                    <h3>Analysis Error</h3>
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

        <div class="panel" style="margin-top: 20px;">
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
