const API_BASE = "";

function getAccessToken() {
  return localStorage.getItem("co2_access_token");
}

function authHeaders(extra = {}) {
  const token = getAccessToken();
  return { ...extra, ...(token ? { Authorization: `Bearer ${token}` } : {}) };
}

function readStoredUser() {
  try { return JSON.parse(localStorage.getItem("co2_user") || "{}"); }
  catch (_) { return {}; }
}

function logout() {
  localStorage.removeItem("co2_access_token");
  localStorage.removeItem("co2_user");
  window.location.replace("/");
}
window.logout = logout;

function safeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"
  }[c]));
}

function currentPage() {
  const path = location.pathname.split("/").filter(Boolean).pop();
  return path || "index.html";
}

function renderAppHeader(user, page) {
  const host = document.getElementById("app-header");
  if (!host) return;

  const isAdmin = user.role === "admin";

  // These are the ONLY primary navigation links for each role.
  const links = isAdmin
    ? [
        ["admin.html", "Train"],
        ["prediction.html", "Predict"],
        ["history.html", "History"]
      ]
    : [
        ["prediction.html", "Prediction & SHAP"],
        ["batch.html", "Batch Prediction"],
        ["methodology.html", "Methodology"],
        ["history.html", "My History"]
      ];

  host.innerHTML = `
    <div class="header-inner" data-role="${isAdmin ? "admin" : "user"}">
      <a class="app-brand" href="/" aria-label="CarbonLab home">
        <span class="brand-symbol">C₂</span>
        <span class="brand-name">CarbonLab</span>
      </a>
      <nav class="app-nav" aria-label="Main navigation">
        ${links.map(([href, label]) =>
          `<a href="${href}" ${page === href ? 'class="active" aria-current="page"' : ""}>${label}</a>`
        ).join("")}
      </nav>
      <div class="account-actions">
        <span class="account-name" title="Signed in as ${safeText(user.username)}">${safeText(user.username)}</span>
        <button class="logout-button" type="button" id="logout-button">Log out</button>
      </div>
      <button class="mobile-nav-toggle" type="button" aria-label="Toggle navigation" aria-expanded="false">☰</button>
    </div>`;

  host.querySelector("#logout-button")?.addEventListener("click", logout);
  const toggle = host.querySelector(".mobile-nav-toggle");
  toggle?.addEventListener("click", () => {
    const nav = host.querySelector(".app-nav");
    const open = nav.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });
}

async function guardAndRender() {
  const page = currentPage();
  const publicPages = ["index.html", "login.html"];
  if (publicPages.includes(page)) return;

  const token = getAccessToken();
  if (!token) {
    location.replace("/login.html");
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/api/auth/me`, {
      headers: authHeaders()
    });
    if (!response.ok) throw new Error("Authentication expired");

    const user = await response.json();
    localStorage.setItem("co2_user", JSON.stringify(user));

    const isAdmin = user.role === "admin";
    const adminOnlyPages = ["admin.html"];
    const userOnlyPages = ["batch.html", "methodology.html"];

    if (adminOnlyPages.includes(page) && !isAdmin) {
      location.replace("/prediction.html");
      return;
    }

    if (userOnlyPages.includes(page) && isAdmin) {
      location.replace("/admin.html");
      return;
    }

    renderAppHeader(user, page);

    if (page === "prediction.html") loadActiveModel();
  } catch (_) {
    localStorage.removeItem("co2_access_token");
    localStorage.removeItem("co2_user");
    location.replace("/login.html");
  }
}

async function loadActiveModel() {
  const target = document.getElementById("active-model");
  if (!target) return;

  try {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) throw new Error("Unavailable");
    const data = await response.json();
    target.textContent = `Active model: ${data.active_model_version || "Available"}`;
    target.title = data.model_strategy || "Currently active prediction model";
  } catch (_) {
    target.textContent = "Active model information unavailable";
  }
}

document.addEventListener("DOMContentLoaded", guardAndRender);
