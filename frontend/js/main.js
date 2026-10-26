/**
 * Shared API and navigation configuration
 */
const API_BASE = "";

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
    checkBackendHealth();
});
