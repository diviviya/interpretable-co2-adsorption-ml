// =====================================================================
// common.js — small helpers every page uses, plus the top navigation.
// =====================================================================

const $ = (id) => document.getElementById(id);

// Format a number without long tails: 79.10499999 -> "79.105"
function fmt(n, digits = 3) {
  if (n === null || n === undefined || Number.isNaN(n)) return "–";
  return Number(Number(n).toFixed(digits)).toString();
}

function fmtSigned(n, digits = 3) {
  return (n > 0 ? "+" : n < 0 ? "−" : "") + fmt(Math.abs(n), digits);
}

function fmtDate(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// Make text safe to put inside innerHTML (stops CSV contents running as HTML).
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Read a CSV string into { columns, rows }. Handles quoted values with commas.
function parseCSV(text) {
  const records = [];
  let row = [], cell = "", inQuotes = false;
  text = text.replace(/^﻿/, ""); // drop Excel's invisible BOM character
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') inQuotes = false;
      else cell += ch;
    } else if (ch === '"') inQuotes = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); records.push(row); row = []; cell = "";
    } else cell += ch;
  }
  if (cell !== "" || row.length) { row.push(cell); records.push(row); }

  const nonEmpty = records.filter((r) => r.some((c) => c.trim() !== ""));
  if (!nonEmpty.length) return { columns: [], rows: [] };
  const columns = nonEmpty[0].map((c) => c.trim());
  const rows = nonEmpty.slice(1).map((r) => {
    const obj = {};
    columns.forEach((c, i) => { obj[c] = (r[i] ?? "").trim(); });
    return obj;
  });
  return { columns, rows };
}

// Turn rows back into CSV text (for downloads).
function toCSV(columns, rows) {
  const quote = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  return [columns.map(quote).join(","), ...rows.map((r) => columns.map((c) => quote(r[c])).join(","))].join("\n");
}

function downloadText(filename, text, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Basic file checks before sending anything. The backend still validates properly.
function checkCsvFile(file) {
  if (!file) return "Choose a CSV file first.";
  if (!/\.csv$/i.test(file.name)) return "The file must be a .csv file.";
  if (file.size === 0) return "The file is empty.";
  if (file.size > CONFIG.MAX_UPLOAD_MB * 1024 * 1024) return `The file is larger than ${CONFIG.MAX_UPLOAD_MB} MB.`;
  return "";
}

// ---------- Navigation bar, demo banner and backend status ----------
const NAV_LINKS = [
  { href: "predict.html", label: "Predict",        page: "predict" },
  { href: "selector.html", label: "Selection Gate", page: "selector" },
  { href: "batch.html",   label: "Batch",          page: "batch" },
  { href: "history.html", label: "History",        page: "history" },
  { href: "about.html",   label: "About",          page: "about" },
];

function renderNav(current) {
  const host = $("site-nav");
  if (!host) return;
  const links = NAV_LINKS.map((l) =>
    `<a href="${l.href}"${l.page === current ? ' aria-current="page"' : ""}>${l.label}</a>`).join("");
  host.innerHTML = `
    <div class="wrap nav-inner">
      <a class="brand" href="index.html"><span class="brand-mark">CO₂</span> Uptake Predictor</a>
      <nav class="nav-links" aria-label="Main">${links}</nav>
      <div class="nav-right">
        <span class="status" id="api-status"><span class="dot"></span><span class="status-text">Checking…</span></span>
        <a class="admin-link${current === "admin" ? " is-current" : ""}" href="admin.html">Admin</a>
      </div>
    </div>
    ${CONFIG.USE_MOCK ? '<div class="demo-banner">Demo mode: results on this site are mock data, not the real model. Set <code>USE_MOCK: false</code> in <code>js/config.js</code> once the backend is running.</div>' : ""}`;
  checkBackend();
}

async function checkBackend() {
  const el = $("api-status");
  if (!el) return;
  const text = el.querySelector(".status-text");
  if (CONFIG.USE_MOCK) {
    el.dataset.state = "mock";
    text.textContent = "Demo data";
    return;
  }
  try {
    await API.health();
    el.dataset.state = "ok";
    text.textContent = "Backend online";
  } catch {
    el.dataset.state = "down";
    text.textContent = "Backend offline";
  }
}
