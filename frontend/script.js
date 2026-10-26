const form = document.getElementById("form");
const result = document.getElementById("result");
const predictBtn = document.getElementById("predict");
const exampleBtn = document.getElementById("example");
const clearBtn = document.getElementById("clear");

const example = {
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
  exampleBtn.onclick = () => {
    Object.entries(example).forEach(([k, v]) => {
      if (form.elements[k]) form.elements[k].value = v;
    });
  };
}

if (clearBtn) {
  clearBtn.onclick = () => {
    form.reset();
    result.innerHTML = `
      <div>
        <div class="icon">↗</div>
        <h3>Your prediction will appear here</h3>
        <p>Fill the features and click the prediction button.</p>
      </div>
    `;
  };
}

form.onsubmit = async function (e) {
  e.preventDefault();

  const data = Object.fromEntries(new FormData(form));

  for (let key in data) {
    data[key] = Number(data[key]);
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting & Explaining...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    const resultData = await response.json();

    if (!response.ok) {
      throw new Error(resultData.detail || "Prediction failed. Check your inputs.");
    }

    const shapList = resultData.shap_contributions || [];
    const maxVal = Math.max(...shapList.map(x => Math.abs(x.shap_value)), 0.001);

    result.innerHTML = `
      <div style="width: 100%;">
        <div class="prediction-box">
          <div class="prediction-num">${Number(resultData.predicted_co2_uptake_mmol_g).toFixed(3)}</div>
          <div class="prediction-unit">mmol/g &bull; Predicted CO₂ Uptake</div>
        </div>

        ${shapList.length > 0 ? `
          <div class="shap-card">
            <div class="shap-header">
              <div class="shap-title">
                <span>🔍</span> SHAP Attribution (Why?)
              </div>
              <div class="shap-base">Baseline: ${resultData.base_value || '3.49'} mmol/g</div>
            </div>
            <p style="font-size: 11px; color: var(--muted); margin-bottom: 10px; line-height: 1.4;">
              Feature impact breakdown (pushing prediction above or below baseline):
            </p>
            <div class="shap-list">
              ${shapList.slice(0, 5).map(c => {
                const pct = Math.min(100, Math.max(12, Math.round((Math.abs(c.shap_value) / maxVal) * 100)));
                const isPos = c.shap_value >= 0;
                return `
                  <div class="shap-item">
                    <div class="shap-row">
                      <span class="shap-feat">${c.feature}: <b>${c.input_value}</b></span>
                      <span class="shap-val ${isPos ? 'shap-pos' : 'shap-neg'}">${c.formatted_shap} mmol/g</span>
                    </div>
                    <div class="shap-bar-bg">
                      <div class="shap-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${pct}%;"></div>
                    </div>
                  </div>
                `;
              }).join('')}
            </div>
          </div>
        ` : ''}
      </div>
    `;
  } catch (error) {
    result.innerHTML = `<p style="color:#d9534f; padding: 15px;">${error.message || "Could not connect to the backend."}</p>`;
  }

  predictBtn.disabled = false;
  predictBtn.textContent = "Predict CO₂ uptake →";
};
