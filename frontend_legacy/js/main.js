/**
 * Shared API and navigation configuration
 */
const API_BASE = "";

function getAccessToken() {
    return localStorage.getItem("co2_access_token");
}

function authHeaders(extra = {}) {
    const token = getAccessToken();
    return { ...extra, ...(token ? { Authorization: `Bearer ${token}` } : {}) };
}

function requireLogin() {
    const protectedPages = ["prediction.html", "history.html"];
    const currentPage = window.location.pathname.split("/").pop();
    if (protectedPages.includes(currentPage) && !getAccessToken()) {
        const returnTo = `${currentPage}${window.location.hash || ""}`;
        window.location.replace(`login.html?next=${encodeURIComponent(returnTo)}`);
        return false;
    }
    return true;
}

function logout() {
    localStorage.removeItem("co2_access_token");
    localStorage.removeItem("co2_user");
    window.location.href = "login.html";
}

// Check backend status on load
async function checkBackendHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const badge = document.getElementById("backend-status");
        if (badge) {
            badge.innerHTML = `<i></i> Online &bull; ${data.model_strategy || "Ensemble"}`;
        }
    } catch (e) {
        const badge = document.getElementById("backend-status");
        if (badge) {
            badge.innerHTML = `<i style="background:#e53935"></i> Offline`;
            badge.style.color = "#c62828";
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    if (!requireLogin()) return;
    checkBackendHealth();
});
