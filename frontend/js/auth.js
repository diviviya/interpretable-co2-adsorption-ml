const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");
async function readResponse(response) { const data = await response.json().catch(() => ({})); if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`); return data; }
if (loginForm) loginForm.addEventListener("submit", async (event) => {
  event.preventDefault(); const status = document.getElementById("login-status"); const button = loginForm.querySelector("button[type=submit]");
  status.textContent = "Signing in…"; button.disabled = true;
  try {
    const payload = Object.fromEntries(new FormData(loginForm));
    const data = await readResponse(await fetch("/api/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }));
    localStorage.setItem("co2_access_token", data.access_token); localStorage.setItem("co2_user", JSON.stringify(data.user));
    location.replace(data.user.role === "admin" ? "/admin.html" : "/prediction.html");
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
});
if (registerForm) registerForm.addEventListener("submit", async (event) => {
  event.preventDefault(); const status = document.getElementById("register-status"); const button = registerForm.querySelector("button[type=submit]");
  status.textContent = "Creating account…"; button.disabled = true;
  try {
    const payload = Object.fromEntries(new FormData(registerForm));
    const data = await readResponse(await fetch("/api/auth/register", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }));
    status.textContent = `Account created for ${data.username}. Sign in to continue.`; registerForm.reset();
    document.querySelector('[data-auth-tab="login"]')?.click();
    const loginName = loginForm?.elements.namedItem("username"); if (loginName) loginName.value = payload.username;
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
});
