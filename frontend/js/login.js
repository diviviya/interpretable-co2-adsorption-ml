// =====================================================================
// login.js — used by login.html (users: log in or sign up) and
// admin-login.html (admins only). <body data-auth="user|admin"> says which.
// The server checks everything; this page only collects the details.
// =====================================================================

const isAdminPage = document.body.dataset.auth === "admin";
let mode = "login";   // "login" or "signup" (signup exists on the user page only)

$("brand").innerHTML = `${BRAND_MARK} CO₂ Uptake Predictor`;
drawLattice($("side-lattice"), { cols: 7, rows: 6, size: 30, onDark: true, seed: isAdminPage ? 23 : 5 });

// Already logged in with the right kind of account? Skip this page.
const existing = Auth.user();
if (existing && (!isAdminPage || existing.role === "admin")) {
  location.replace(nextPage(isAdminPage ? "admin.html" : "predict.html"));
}

function showAlerts(items) {
  $("auth-alerts").innerHTML = items.map(([kind, t]) => `<li class="${kind}">${escapeHtml(t)}</li>`).join("");
}

// ---------- Log in / Create account tabs (user page) ----------
function setMode(next) {
  mode = next;
  const signup = mode === "signup";
  $("tab-login").setAttribute("aria-selected", String(!signup));
  $("tab-signup").setAttribute("aria-selected", String(signup));
  $("auth-title").textContent = signup ? "Create your account" : "Welcome back";
  $("auth-sub").textContent = signup
    ? "Free. Your predictions are saved to your own history."
    : "Log in to make predictions and see your history.";
  $("auth-btn").textContent = signup ? "Create account" : "Log in";
  $("confirm-box").hidden = !signup;
  $("username-hint").hidden = !signup;
  $("password-hint").hidden = !signup;
  $("password").autocomplete = signup ? "new-password" : "current-password";
  showAlerts([]);
}

if (!isAdminPage) {
  $("tab-login").addEventListener("click", () => setMode("login"));
  $("tab-signup").addEventListener("click", () => setMode("signup"));
  if (new URLSearchParams(location.search).get("mode") === "signup") setMode("signup");
}

// ---------- Submit ----------
$("auth-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = $("username").value.trim();
  const password = $("password").value;

  // Quick checks so obvious mistakes don't need a round trip.
  // The server repeats every check.
  if (!username || !password) { showAlerts([["error", "Enter your username and password."]]); return; }
  if (mode === "signup") {
    if (!/^[A-Za-z0-9_.-]{3,30}$/.test(username)) { showAlerts([["error", "Username must be 3–30 characters: letters, numbers, _ . or -"]]); return; }
    if (password.length < 8) { showAlerts([["error", "Password must be at least 8 characters."]]); return; }
    if (password !== $("confirm").value) { showAlerts([["error", "The two passwords don't match."]]); return; }
  }

  const btn = $("auth-btn");
  const label = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> ${mode === "signup" ? "Creating account…" : "Logging in…"}`;
  try {
    const res = mode === "signup"
      ? await API.signup(username, password)
      : await API.login(username, password, isAdminPage);
    Auth.set({ token: res.token, user: res.user });
    location.replace(nextPage(isAdminPage ? "admin.html" : "predict.html"));
  } catch (err) {
    showAlerts([["error", err.message]]);
    btn.disabled = false;
    btn.textContent = label;
  }
});

$("username").focus();
