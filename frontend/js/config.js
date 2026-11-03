// =====================================================================
// config.js — every setting the frontend needs, in one place.
// =====================================================================

const CONFIG = {
  // true  = use the fake responses in mock.js (no backend needed)
  // false = talk to the real FastAPI backend at API_BASE
  USE_MOCK: false,

  // Where the FastAPI backend runs. Change this when it is deployed.
  API_BASE: "http://127.0.0.1:8000",

  // Mock mode only: real project files the mocks read from your repo.
  MOCK_FEATURE_INFO_URL: "../artifacts/feature_info.json",
  MOCK_IMPORTANCE_URL: "../results/08_shap_importance.csv",

  REQUEST_TIMEOUT_MS: 30000, // give up on a request after 30 s
  MAX_UPLOAD_MB: 5,          // quick size check before uploading a CSV
};

// The column the model predicts.
const TARGET = "CO2 uptake (mmol/g)";

// The 10 model inputs.
// column : the exact name used in the dataset AND sent to the backend
// id     : a short name used only for HTML element ids on the page
const FIELDS = [
  { column: "Surface Area (m2/g)",      id: "sa",   label: "Surface area",      unit: "m²/g",  group: "structure",   required: true },
  { column: "Total Pore Volume(cm3/g)", id: "tpv",  label: "Total pore volume", unit: "cm³/g", group: "structure",   required: false },
  { column: "Micropore Volume (cm3/g)", id: "mpv",  label: "Micropore volume",  unit: "cm³/g", group: "structure",   required: false },
  { column: "C (%)",                    id: "c",    label: "Carbon",            unit: "%",     group: "composition", required: true, percent: true },
  { column: "H (%)",                    id: "h",    label: "Hydrogen",          unit: "%",     group: "composition", required: true, percent: true },
  { column: "N (%)",                    id: "n",    label: "Nitrogen",          unit: "%",     group: "composition", required: true, percent: true },
  { column: "O (%)",                    id: "o",    label: "Oxygen",            unit: "%",     group: "composition", required: true, percent: true },
  { column: "S (%)",                    id: "s",    label: "Sulfur",            unit: "%",     group: "composition", required: true, percent: true },
  { column: "Temp (°C)",                id: "temp", label: "Temperature",       unit: "°C",    group: "conditions",  required: true, allowNegative: true },
  { column: "Pressure (bar)",           id: "pres", label: "Pressure",          unit: "bar",   group: "conditions",  required: true },
];

const FEATURE_COLUMNS = FIELDS.map((f) => f.column);
