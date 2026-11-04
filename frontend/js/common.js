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

// ---------- Login state ----------
// After logging in, the server gives this browser a session token. We keep
// it (and the user's name and role) in localStorage so every page can send
// it. Logging out deletes it here AND ends the session on the server.
const Auth = {
  KEY: "co2_session",
  get()   { try { return JSON.parse(localStorage.getItem(this.KEY)); } catch { return null; } },
  set(s)  { try { localStorage.setItem(this.KEY, JSON.stringify(s)); } catch { /* storage blocked */ } },
  clear() { try { localStorage.removeItem(this.KEY); } catch { /* storage blocked */ } },
  token() { const s = this.get(); return s ? s.token : ""; },
  user()  { const s = this.get(); return s ? s.user : null; },
};

// Which login page this page sends people to: <body data-protected="admin">
// pages use the admin login, everything else the normal one.
function loginPageFor(role) {
  const here = location.pathname.split("/").pop() || "index.html";
  return `${role === "admin" ? "admin-login.html" : "login.html"}?next=${encodeURIComponent(here)}`;
}

// Where to go after logging in: the page that sent you here, but only a
// plain page name from this site (never an outside link).
function nextPage(fallback) {
  const next = new URLSearchParams(location.search).get("next") || "";
  return /^[a-z-]+\.html$/.test(next) ? next : fallback;
}

// Page guard. Pages with <body data-protected="user|admin"> need a login.
// 1) No saved session -> straight to the login page (before anything loads).
// 2) Otherwise ask the server "is this session still valid?" in the
//    background; api.js sends people to the login page if it isn't.
(function guardPage() {
  const need = document.body && document.body.dataset.protected;
  if (!need) return;
  const user = Auth.user();
  if (!user || (need === "admin" && user.role !== "admin")) {
    location.replace(loginPageFor(need));
    return;
  }
  API.me()
    .then(({ user: fresh }) => {
      Auth.set({ ...Auth.get(), user: fresh });
      if (need === "admin" && fresh.role !== "admin") location.replace(loginPageFor("admin"));
    })
    .catch(() => { /* 401 is handled in api.js; offline: keep the saved user */ });
})();

async function logout() {
  try { await API.logout(); } catch { /* ending it locally is enough */ }
  Auth.clear();
  location.href = "index.html";
}

// ---------- Navigation bar, demo banner and backend status ----------
const NAV_LINKS = [
  { href: "predict.html", label: "Predict",  page: "predict", needs: "user" },
  { href: "batch.html",   label: "Batch",    page: "batch",   needs: "user" },
  { href: "history.html", label: "History",  page: "history", needs: "user" },
  { href: "about.html",   label: "About",    page: "about" },
  { href: "admin.html",   label: "Admin",    page: "admin",   needs: "admin" },
];

// The brand mark: one hexagon of a carbon lattice with a CO₂ molecule inside.
const BRAND_MARK = `<svg class="brand-mark" viewBox="0 0 26 26" aria-hidden="true">
  <path d="M13 1.5 23 7.25v11.5L13 24.5 3 18.75V7.25z" fill="none" stroke="#7cc6dc" stroke-width="2"/>
  <line x1="7.5" y1="13" x2="18.5" y2="13" stroke="#fff" stroke-width="1.8"/>
  <circle cx="13" cy="13" r="2.6" fill="#fff"/><circle cx="7.3" cy="13" r="2.2" fill="#ef8a5c"/><circle cx="18.7" cy="13" r="2.2" fill="#ef8a5c"/>
</svg>`;

function renderNav(current) {
  const host = $("site-nav");
  if (!host) return;
  const user = Auth.user();
  const visible = NAV_LINKS.filter((l) => !l.needs || (user && (l.needs === "user" || user.role === "admin")));
  const links = visible.map((l) =>
    `<a href="${l.href}"${l.page === current ? ' aria-current="page"' : ""}>${l.label}</a>`).join("");

  const right = user
    ? `<span class="user-chip">${escapeHtml(user.username)}${user.role === "admin" ? '<span class="role-tag">admin</span>' : ""}</span>
       <button type="button" class="nav-btn" id="logout-nav">Log out</button>`
    : `<a class="nav-btn" href="login.html">Log in</a>
       <a class="nav-btn solid" href="login.html?mode=signup">Create account</a>`;

  host.innerHTML = `
    <div class="wrap nav-inner">
      <a class="brand" href="index.html">${BRAND_MARK} CO₂ Uptake Predictor</a>
      <button type="button" class="nav-toggle" id="nav-toggle" aria-expanded="false" aria-controls="nav-links">Menu</button>
      <nav class="nav-links" id="nav-links" aria-label="Main">${links}</nav>
      <div class="nav-right">
        <span class="status" id="api-status"><span class="dot"></span><span class="status-text">Checking…</span></span>
        ${right}
      </div>
    </div>
    ${CONFIG.USE_MOCK ? '<div class="demo-banner">Demo mode: results on this site are mock data, not the real model. Set <code>USE_MOCK: false</code> in <code>js/config.js</code> once the backend is running.</div>' : ""}`;

  // Phone menu: the links and account buttons fold behind one button.
  $("nav-toggle").addEventListener("click", () => {
    const open = host.classList.toggle("is-open");
    $("nav-toggle").setAttribute("aria-expanded", String(open));
    $("nav-toggle").textContent = open ? "Close" : "Menu";
  });
  if (user) $("logout-nav").addEventListener("click", logout);
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
    text.textContent = "Online";
  } catch {
    el.dataset.state = "down";
    text.textContent = "Backend offline";
  }
}

// ---------- Carbon lattice illustration (landing + login pages) ----------
// Draws a honeycomb of carbon atoms (like a sheet of porous carbon). Some
// hexagons are "pores", and CO₂ molecules (O=C=O) settle into a few of them
// once when the page opens. Purely decorative: aria-hidden.
function drawLattice(svg, { cols = 9, rows = 6, size = 26, onDark = false, seed = 7 } = {}) {
  let s = seed;
  const rand = () => ((s = (s * 9301 + 49297) % 233280) / 233280);   // repeatable "random"
  const w = Math.sqrt(3) * size, h = 1.5 * size;
  const W = cols * w + w / 2 + 8, H = rows * h + size / 2 + 8;
  svg.setAttribute("viewBox", `0 0 ${W.toFixed(0)} ${H.toFixed(0)}`);
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("lattice");
  if (onDark) svg.classList.add("on-dark");

  const hexPoints = (cx, cy) => Array.from({ length: 6 }, (_, k) => {
    const a = (Math.PI / 180) * (60 * k - 30);
    return [cx + size * Math.cos(a), cy + size * Math.sin(a)];
  });

  let pores = "", bonds = "", atoms = new Map(), mols = "", i = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cx = 4 + w / 2 + c * w + (r % 2 ? w / 2 : 0), cy = 4 + size + r * h;
      const pts = hexPoints(cx, cy);
      const isPore = rand() < 0.22;
      if (isPore) {
        pores += `<polygon class="pore" points="${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ")}"/>`;
        if (rand() < 0.75) {
          const tilt = Math.round(rand() * 160 - 80), L = size * 0.5;
          const dx = Math.round(rand() * 80 - 40);
          mols += `<g class="mol" style="--i:${i++};--dx:${dx}px"><g transform="translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${tilt})">
            <line class="mol-bond" x1="${-L}" y1="0" x2="${L}" y2="0"/>
            <circle class="mol-o" cx="${-L}" cy="0" r="${(size * 0.22).toFixed(1)}"/>
            <circle class="mol-o" cx="${L}" cy="0" r="${(size * 0.22).toFixed(1)}"/>
            <circle class="mol-c" cx="0" cy="0" r="${(size * 0.2).toFixed(1)}"/></g></g>`;
        }
      }
      bonds += `<polygon class="bond" points="${pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ")}"/>`;
      pts.forEach(([x, y]) => atoms.set(`${x.toFixed(0)},${y.toFixed(0)}`, [x, y]));
    }
  }
  const dots = [...atoms.values()].map(([x, y]) => `<circle class="atom" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.6"/>`).join("");
  svg.innerHTML = pores + bonds + dots + mols;
}
