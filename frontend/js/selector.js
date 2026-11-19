// =====================================================================
// selector.js — the "Find materials" page.
//   1. read the customer's conditions, ask the backend to rank real
//      activated carbons from each biomass source
//   2. show them grouped by source, with prediction ranges and the value
//      the paper measured
//   3. "Explore" one carbon: change its properties, or sweep one property
//      to find the value with the highest predicted uptake
// Every call goes through api.js (API.selection...).
// =====================================================================

renderNav("selector");

// Technical columns and wording are for admins only; normal users get the simple view.
const IS_ADMIN = !!(Auth.user() && Auth.user().role === "admin");
document.body.classList.toggle("is-admin", IS_ADMIN);

let catalogue = null;   // answer of /api/selection/carbons
let lastScreen = null;  // answer of the last /api/selection/screen
let chosen = null;      // the carbon open in the Explore panel

// Properties the user can change in Explore, in display order.
const MATERIAL_FIELDS = [
  { key: "surface_area",      label: "Surface area",      unit: "m²/g",  help: "How much surface the carbon offers for CO₂ to stick to." },
  { key: "micropore_volume",  label: "Tiny-pore volume",  unit: "cm³/g", help: "Volume of the tiniest pores, which hold CO₂ best." },
  { key: "total_pore_volume", label: "Total pore volume", unit: "cm³/g", help: "All the empty space inside the carbon." },
  { key: "carbon",            label: "Carbon content",    unit: "%",     help: "Share of carbon, by weight." },
  { key: "hydrogen",          label: "Hydrogen content",  unit: "%",     help: "Share of hydrogen, by weight." },
  { key: "nitrogen",          label: "Nitrogen content",  unit: "%",     help: "Share of nitrogen, by weight." },
  { key: "oxygen",            label: "Oxygen content",    unit: "%",     help: "Share of oxygen, by weight." },
];
const SWEEP_OPTIONS = [
  ...MATERIAL_FIELDS,
  { key: "temperature", label: "Temperature", unit: "°C" },
  { key: "pressure",    label: "Pressure",    unit: "bar" },
];

const VERDICT = {
  meets:      ["Good match",     "ok",   "The best estimate reaches your target. The real value could be a little lower, so check the range."],
  borderline: ["Maybe",          "warn", "The best estimate is just below your target, but the likely range still reaches it."],
  unlikely:   ["Unlikely match", "bad",  "Even the top of the likely range falls short of your target."],
};

// ---------- small helpers ----------
const chip = (text, kind, title = "") =>
  `<span class="chip ${kind}"${title ? ` title="${escapeHtml(title)}"` : ""}>${escapeHtml(text)}</span>`;

const rangeText = (iv) => (iv ? `${fmt(iv.lower, 2)} – ${fmt(iv.upper, 2)}` : "");

function setAlerts(listId, items, kind = "error") {
  const box = $(listId);
  box.innerHTML = items.map((t) => `<li class="${kind}">${escapeHtml(t)}</li>`).join("");
}

function setBusy(btn, busy, idleText) {
  btn.disabled = busy;
  btn.textContent = busy ? "Working…" : idleText;
}

// A link is only used if it is a plain https address.
const safeUrl = (u) => (typeof u === "string" && u.startsWith("https://") ? u : "");

function citationHtml(sourceId) {
  const s = catalogue && catalogue.sources[sourceId];
  if (!s) return "";
  const url = safeUrl(s.url);
  const doi = url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener">doi:${escapeHtml(s.doi)}</a>` : escapeHtml(s.doi);
  return `${escapeHtml(s.citation)} ${doi} · ${escapeHtml(s.tables)}`;
}

// ---------- start-up ----------
async function init() {
  try {
    catalogue = await API.selectionCarbons();
  } catch (err) {
    setAlerts("find-alerts", [`Couldn't load the material list. (${err.message})`]);
    $("find-btn").disabled = true;
    return;
  }

  // biomass-source dropdown
  $("in-feed").innerHTML = `<option value="all">All sources</option>` +
    catalogue.feedstocks.map((f) => `<option value="${escapeHtml(f.id)}">${escapeHtml(f.name)}</option>`).join("");

  // training-range hints
  const tr = catalogue.training_range || {};
  if (tr.temperature) $("hint-temp").textContent = `Room temperature is about 25 °C. Most reliable between ${fmt(tr.temperature.min)} and ${fmt(tr.temperature.max)} °C.`;
  if (tr.pressure)    $("hint-pres").textContent = `1 bar is normal air pressure. Most reliable between ${fmt(tr.pressure.min)} and ${fmt(tr.pressure.max)} bar.`;

  // property list for the sweep
  $("sweep-param").innerHTML = SWEEP_OPTIONS.map(
    (o) => `<option value="${o.key}">${escapeHtml(o.label)} (${escapeHtml(o.unit)})</option>`).join("");

  // sources list at the bottom
  $("sources-list").innerHTML = Object.keys(catalogue.sources)
    .map((id) => `<li>${citationHtml(id)}</li>`).join("");
}

// ---------- 1) find materials ----------
$("find-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  setAlerts("find-alerts", []);

  const t = $("in-temp").value.trim();
  const p = $("in-pres").value.trim();
  const target = $("in-target").value.trim();
  const errors = [];
  if (t === "" || Number.isNaN(Number(t))) errors.push("Enter a temperature in °C.");
  if (p === "" || Number.isNaN(Number(p))) errors.push("Enter a pressure in bar.");
  if (target !== "" && (Number.isNaN(Number(target)) || Number(target) <= 0)) errors.push("Target uptake must be a number above 0, or left empty.");
  if (errors.length) return setAlerts("find-alerts", errors);

  setBusy($("find-btn"), true, "Find materials");
  try {
    lastScreen = await API.selectionScreen({
      temperature: Number(t),
      pressure: Number(p),
      target_uptake: target === "" ? null : Number(target),
      feedstock: $("in-feed").value,
    });
    renderResults(lastScreen);
  } catch (err) {
    setAlerts("find-alerts", [err.message]);
  } finally {
    setBusy($("find-btn"), false, "Find materials");
  }
});

function renderResults(res) {
  $("results").hidden = false;
  $("explore").hidden = true;
  setAlerts("result-warnings", res.warnings || [], "warn");

  $("result-intro").innerHTML = summaryHtml(res);

  $("groups").innerHTML = res.feedstocks.map(groupHtml).join("");
}

// One short, plain answer shown above the list.
function summaryHtml(res) {
  const all = [];
  for (const g of res.feedstocks) for (const c of g.carbons) all.push({ ...c, feed: g.name });
  all.sort((a, b) => b.predicted - a.predicted);
  const name = (c) => `<strong>${escapeHtml(c.feed)}, ${escapeHtml(c.label)}</strong> (${fmt(c.predicted, 2)} mmol/g)`;
  const top = (list) => list.slice(0, 3).map(name).join("; ");
  const cond = res.conditions;
  const where = `at ${fmt(cond.temperature)} °C and ${fmt(cond.pressure)} bar`;
  const t = res.target_uptake;
  if (!t) {
    return `Highest predicted CO₂ uptake ${where}: ${name(all[0])}. Enter the amount you need to see which carbons are a good match.`;
  }
  const good = all.filter((c) => c.verdict === "meets");
  const maybe = all.filter((c) => c.verdict === "borderline");
  if (good.length) {
    return `${good.length} of ${all.length} carbons are a good match for ${fmt(t)} mmol/g ${where}. Top picks: ${top(good)}.`;
  }
  if (maybe.length) {
    return `No carbon\u2019s best estimate reaches ${fmt(t)} mmol/g ${where}, but ${maybe.length} come close and might. Best chances: ${top(maybe)}.`;
  }
  return `None of the carbons is likely to reach ${fmt(t)} mmol/g ${where}. The highest predicted is ${name(all[0])}. Try a lower target or different conditions.`;
}

function groupHtml(g) {
  const best = g.best;
  const v = best.verdict ? VERDICT[best.verdict] : null;
  const rows = g.carbons.map(rowHtml).join("");
  const sourceIds = [...new Set(g.carbons.map((c) => c.source))];

  return `
  <article class="card feed">
    <div class="feed-head">
      <div>
        <h2>${escapeHtml(g.name)}</h2>
        <p class="sub" style="margin:0">${escapeHtml(g.type)} · ${g.carbons.length} carbon${g.carbons.length === 1 ? "" : "s"} from published papers</p>
      </div>
      <div class="feed-best">
        <span class="big">${fmt(best.predicted, 2)}</span> <span class="muted">mmol/g, best in this group</span><br>
        <span class="muted">${best.interval ? `likely between ${rangeText(best.interval)}` : ""}</span>
        ${v ? chip(v[0], v[1], v[2]) : ""}
      </div>
    </div>
    <p class="muted" style="margin:10px 0 12px">Depending on how it is made, the predicted CO₂ uptake of ${escapeHtml(g.name.toLowerCase())} carbons ranges from <strong>${fmt(g.spread.min, 2)}</strong> to <strong>${fmt(g.spread.max, 2)}</strong> mmol/g. How the carbon is made matters as much as which biomass is used.</p>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Carbon and how it was made</th>
        <th class="num tech" title="Surface available for CO₂ to stick to, per gram of carbon.">Surface area (m²/g)</th>
        <th class="num tech" title="Volume of the tiniest pores, which hold CO₂ best.">Tiny-pore volume (cm³/g)</th>
        <th class="num" title="How much CO₂ one gram of the carbon is expected to hold, in millimoles. 1 mmol/g is about 44 g of CO₂ per kg of carbon.">CO₂ it can hold (mmol/g)</th>
        <th class="num tech" title="What the authors measured in the lab at 25 °C and 1 bar.">Measured in paper (mmol/g)</th>
        <th class="tech" title="Whether the paper reported all the main properties the model needs.">Data completeness</th><th></th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="src"><strong>Sources:</strong> ${sourceIds.map(citationHtml).join("<br>")}</p>
  </article>`;
}


// Turns the paper's short recipe text into plain language (display only; data is unchanged).
const CHEM_NAMES = { PO: "potassium oxalate", NaNH2: "sodium amide", KOH: "KOH" };
function friendlyActivation(text) {
  if (!text) return "";
  const parts = [];
  if (/potassium oxalate/i.test(text)) parts.push("Chemical activation with potassium oxalate");
  else if (/sodium amide/i.test(text)) parts.push("Nitrogen added using sodium amide");
  else if (/^Chemical activation with (.+)$/i.test(text.split(",")[0].trim())) parts.push(text.split(",")[0].trim());
  else if (/^Physical activation with (.+)$/i.test(text.split(",")[0].trim())) parts.push(text.split(",")[0].trim().replace(/with (.+)$/i, (m, g) => `with ${g} (no chemical added)`));
  else return text;
  const ratio = text.match(/\(([A-Za-z0-9]+):([A-Za-z ]+?)\s+(\d+):(\d+)\)/);
  if (ratio) {
    const chem = CHEM_NAMES[ratio[1]] || ratio[1];
    parts.push(`${ratio[3]} g ${chem} for every ${ratio[4]} g ${ratio[2].trim().toLowerCase()}`);
  }
  const temp = text.match(/(\d{3,4})\s*°C/);
  if (temp) parts.push(`heated to ${temp[1]} °C`);
  if (/compacted pellet/i.test(text)) parts.push("pressed into pellets first");
  else if (/powder mix/i.test(text)) parts.push("mixed as loose powder");
  return parts.join(" · ");
}

function rowHtml(c) {
  const v = c.verdict ? VERDICT[c.verdict] : null;
  const quality = c.quality.complete
    ? chip("All data reported", "ok", "All the main properties were reported in the paper.")
    : chip("Some data missing", "warn", `Not reported in the paper: ${c.quality.missing.join(", ")}. Typical values from the training data are used for these.`);
  return `
    <tr>
      <td><strong title="Name used in the source paper">${escapeHtml(c.label)}</strong><small title="${escapeHtml(c.activation)}">${escapeHtml(friendlyActivation(c.activation))}</small></td>
      <td class="num tech">${fmt(c.surface_area, 0)}</td>
      <td class="num tech">${c.micropore_volume === null ? "–" : fmt(c.micropore_volume, 2)}</td>
      <td class="num"><strong>${fmt(c.predicted, 2)}</strong><small title="The real value is probably in this range.">${c.interval ? "likely " + rangeText(c.interval) : ""}</small>${v ? chip(v[0], v[1], v[2]) : ""}${c.verdict === "meets" && c.confident === false ? `<small>Could fall a little short</small>` : ""}${c.quality.complete ? "" : `<small class="user-only">Less certain: the paper did not report every detail.</small>`}</td>
      <td class="num tech">${c.reported_uptake_25c_1bar === null ? "–" : fmt(c.reported_uptake_25c_1bar, 2)}</td>
      <td class="tech">${quality}</td>
      <td><button type="button" class="btn ghost" data-explore="${escapeHtml(c.id)}">Try changes</button></td>
    </tr>`;
}

$("groups").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-explore]");
  if (btn) openExplore(btn.dataset.explore);
});

// ---------- 2) explore one carbon ----------
function findCarbon(id) {
  for (const g of lastScreen.feedstocks) for (const c of g.carbons) if (c.id === id) return c;
  return null;
}

function openExplore(id) {
  chosen = findCarbon(id);
  if (!chosen) return;
  const feedName = lastScreen.feedstocks.find((g) => g.id === chosen.feedstock).name;

  $("explore").hidden = false;
  $("explore-title").textContent = `Try changes: ${feedName}, ${chosen.label}`;
  $("explore-sub").innerHTML = `${escapeHtml(friendlyActivation(chosen.activation))}. Change any value to see how the predicted CO₂ uptake changes. Leave a box empty to use a typical value.`;

  $("explore-fields").innerHTML = MATERIAL_FIELDS.map((f) => `
    <div class="field">
      <label for="ex-${f.key}">${f.label} <span class="unit">(${f.unit})</span></label>
      <input id="ex-${f.key}" type="number" step="any" inputmode="decimal" placeholder="typical">
      <small class="hint">${f.help}</small>
    </div>`).join("") + `
    <div class="field"><label for="ex-temperature">Temperature <span class="unit">(°C)</span></label>
      <input id="ex-temperature" type="number" step="any" inputmode="decimal"></div>
    <div class="field"><label for="ex-pressure">Pressure <span class="unit">(bar)</span></label>
      <input id="ex-pressure" type="number" step="any" inputmode="decimal"></div>`;

  resetExploreFields();
  $("explore-result").hidden = true;
  $("sweep-result").innerHTML = "";
  setAlerts("explore-alerts", []);
  recalc();
  $("explore").scrollIntoView({ behavior: "smooth", block: "start" });
}

function resetExploreFields() {
  for (const f of MATERIAL_FIELDS) {
    const v = chosen[f.key];
    $(`ex-${f.key}`).value = v === null || v === undefined ? "" : v;
  }
  $("ex-temperature").value = lastScreen.conditions.temperature;
  $("ex-pressure").value = lastScreen.conditions.pressure;
}

$("reset-btn").addEventListener("click", () => { resetExploreFields(); recalc(); });

// Read the Explore boxes: returns { temperature, pressure, overrides } or an error list.
function readExplore() {
  const errors = [];
  const overrides = {};
  for (const f of MATERIAL_FIELDS) {
    const raw = $(`ex-${f.key}`).value.trim();
    if (raw === "") {
      if (chosen[f.key] !== null) overrides[f.key] = null;   // cleared on purpose -> typical value
      continue;
    }
    const v = Number(raw);
    if (Number.isNaN(v) || v < 0) { errors.push(`${f.label} must be a number, 0 or more.`); continue; }
    overrides[f.key] = v;
  }
  const t = $("ex-temperature").value.trim();
  const p = $("ex-pressure").value.trim();
  if (t === "" || Number.isNaN(Number(t))) errors.push("Enter a temperature.");
  if (p === "" || Number.isNaN(Number(p))) errors.push("Enter a pressure.");
  return { errors, temperature: Number(t), pressure: Number(p), overrides };
}

$("recalc-btn").addEventListener("click", recalc);

async function recalc() {
  if (!chosen) return;
  const r = readExplore();
  if (r.errors.length) return setAlerts("explore-alerts", r.errors);
  setAlerts("explore-alerts", []);
  setBusy($("recalc-btn"), true, "Recalculate");
  try {
    const res = await API.selectionWhatIf({
      carbon_id: chosen.id, temperature: r.temperature, pressure: r.pressure, overrides: r.overrides,
    });
    setAlerts("explore-alerts", res.warnings || [], "warn");
    const box = $("explore-result");
    const reported = chosen.reported_uptake_25c_1bar;
    box.hidden = false;
    box.innerHTML = `
      <p class="label-caps">Predicted CO₂ uptake</p>
      <p><span class="big-num">${fmt(res.predicted, 2)}</span> mmol/g
         <span class="muted">${res.interval ? ` · likely ${rangeText(res.interval)}` : ""}</span></p>
      <p class="muted" style="margin:6px 0 0">${
        Math.abs(res.change) < 0.005
          ? "Same as the carbon with its reported properties."
          : `${fmtSigned(res.change, 2)} mmol/g compared with the same carbon at the same conditions with its reported properties (${fmt(res.baseline.predicted, 2)}).`}
        ${reported !== null ? ` The paper measured ${fmt(reported, 2)} mmol/g at 25 °C and 1 bar.` : ""}</p>`;
  } catch (err) {
    setAlerts("explore-alerts", [err.message]);
  } finally {
    setBusy($("recalc-btn"), false, "Recalculate");
  }
}

// ---------- 3) sweep one property ----------
$("sweep-btn").addEventListener("click", async () => {
  if (!chosen) return;
  const r = readExplore();
  if (r.errors.length) return setAlerts("explore-alerts", r.errors);
  setAlerts("explore-alerts", []);
  const parameter = $("sweep-param").value;
  setBusy($("sweep-btn"), true, "Find the best value");
  try {
    const res = await API.selectionSweep({
      carbon_id: chosen.id, parameter, temperature: r.temperature, pressure: r.pressure,
      overrides: r.overrides, points: 30,
    });
    renderSweep(res);
  } catch (err) {
    $("sweep-result").innerHTML = "";
    setAlerts("explore-alerts", [err.message]);
  } finally {
    setBusy($("sweep-btn"), false, "Find the best value");
  }
});

function renderSweep(res) {
  const opt = SWEEP_OPTIONS.find((o) => o.key === res.parameter) || { label: res.label || res.parameter, unit: "" };
  const tr = res.training_range;
  const pts = res.points;
  const first = pts[0], last = pts[pts.length - 1];
  const flat = Math.abs(res.best.predicted - Math.min(...pts.map((p) => p.predicted))) < 0.05;

  $("sweep-result").innerHTML = `
    <p style="margin:14px 0 4px"><strong>Highest predicted uptake: ${fmt(res.best.predicted, 2)} mmol/g</strong>
      when ${escapeHtml(opt.label.toLowerCase())} is about <strong>${fmt(res.best.value, 3)} ${escapeHtml(opt.unit)}</strong>
      (tested from ${fmt(first.value, 3)} to ${fmt(last.value, 3)} ${escapeHtml(opt.unit)}).</p>
    ${flat ? `<p class="muted" style="margin:0">The curve is almost flat, so this property has little effect for this carbon.</p>` : ""}
    ${chartSvg(pts, res.best, `${opt.label} (${opt.unit})`)}
    <p class="note">This is what the <em>model</em> predicts. It does not mean you can set that value freely: surface area and pore volume, for example, depend on how the carbon is made (the activation route above), and properties change together in real materials.
      ${res.outside_training_range ? "Part of this range is outside the data the model was trained on, so treat that part with extra caution." : ""}</p>`;
}

// A small line chart drawn as SVG (no chart library needed).
function chartSvg(points, best, xLabel) {
  const W = 620, H = 250, m = { l: 52, r: 16, t: 16, b: 44 };
  const xs = points.map((p) => p.value), ys = points.map((p) => p.predicted);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y1 - y0 < 0.1) { y0 -= 0.05; y1 += 0.05; }
  const pad = (y1 - y0) * 0.12; y0 -= pad; y1 += pad;
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);

  const path = points.map((p, i) => `${i ? "L" : "M"}${X(p.value).toFixed(1)},${Y(p.predicted).toFixed(1)}`).join(" ");
  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4;
    grid += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(yv)}" y2="${Y(yv)}" style="stroke:var(--line-soft)"/>
             <text x="${m.l - 8}" y="${Y(yv) + 4}" text-anchor="end" font-size="11" style="fill:var(--slate)">${fmt(yv, 2)}</text>`;
    const xv = x0 + ((x1 - x0) * i) / 4;
    grid += `<text x="${X(xv)}" y="${H - m.b + 16}" text-anchor="middle" font-size="11" style="fill:var(--slate)">${fmt(xv, 3)}</text>`;
  }
  return `<svg class="sweep-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted CO2 uptake against ${escapeHtml(xLabel)}">
    ${grid}
    <path d="${path}" fill="none" stroke-width="2.4" style="stroke:var(--petrol)"/>
    <circle cx="${X(best.value)}" cy="${Y(best.predicted)}" r="5.5" style="fill:var(--down)"/>
    <text x="${(W + m.l - m.r) / 2}" y="${H - 6}" text-anchor="middle" font-size="12" style="fill:var(--graphite)">${escapeHtml(xLabel)}</text>
    <text x="14" y="${(H - m.b + m.t) / 2}" font-size="12" text-anchor="middle" transform="rotate(-90 14 ${(H - m.b + m.t) / 2})" style="fill:var(--graphite)">Predicted uptake (mmol/g)</text>
  </svg>`;
}

init();
