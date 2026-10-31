/**
 * Admin Portal Controller
 * Manages JWT Auth, Dataset Upload, Training Jobs, Model Activation/Rollback, and Audit Logs.
 */

let authToken = localStorage.getItem("co2_auth_token") || null;
let currentUser = null;
let activePollingInterval = null;

// DOM Elements
const authPill = document.getElementById("admin-user-pill");
const btnAuthAction = document.getElementById("btn-auth-action");
const loginBanner = document.getElementById("login-required-banner");
const btnBannerLogin = document.getElementById("btn-banner-login");

const loginModal = document.getElementById("login-modal");
const loginForm = document.getElementById("login-form");
const loginUsername = document.getElementById("login-username");
const loginPassword = document.getElementById("login-password");
const loginError = document.getElementById("login-error");
const btnCancelLogin = document.getElementById("btn-cancel-login");

const confirmModal = document.getElementById("confirm-modal");
const confirmTitle = document.getElementById("confirm-title");
const confirmMessage = document.getElementById("confirm-message");
const btnConfirmCancel = document.getElementById("btn-confirm-cancel");
const btnConfirmOk = document.getElementById("btn-confirm-ok");
let pendingConfirmAction = null;

const heroActiveVersion = document.getElementById("hero-active-version");
const heroActiveR2 = document.getElementById("hero-active-r2");

const datasetForm = document.getElementById("dataset-upload-form");
const datasetInput = document.getElementById("dataset-file-input");
const datasetFeedback = document.getElementById("dataset-feedback");

const btnStartTraining = document.getElementById("btn-start-training");
const trainingStatusTag = document.getElementById("training-status-tag");
const trainingSpinner = document.getElementById("training-spinner");
const trainingLogCard = document.getElementById("training-log-card");
const jobIdDisplay = document.getElementById("job-id-display");
const jobStatusPill = document.getElementById("job-status-pill");
const jobDetailsDisplay = document.getElementById("job-details-display");
const jobMetricsDisplay = document.getElementById("job-metrics-display");

const adminModelsTbody = document.getElementById("admin-models-tbody");
const btnRefreshModels = document.getElementById("btn-refresh-models");
const auditLogsTbody = document.getElementById("audit-logs-tbody");

// -----------------------------------------------------------------------------
// 1. AUTHENTICATION & SESSION MANAGEMENT
// -----------------------------------------------------------------------------
async function checkAuthSession() {
    if (!authToken) {
        setGuestState();
        return;
    }
    try {
        const res = await fetch("/api/auth/me", {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        if (res.ok) {
            currentUser = await res.json();
            authPill.textContent = `Logged in: ${currentUser.username} (${currentUser.role})`;
            authPill.style.background = "#1b5e20";
            btnAuthAction.textContent = "Logout";
            loginBanner.style.display = "none";
        } else {
            logout();
        }
    } catch (e) {
        logout();
    }
}

function setGuestState() {
    currentUser = null;
    authToken = null;
    localStorage.removeItem("co2_auth_token");
    authPill.textContent = "Guest Access";
    authPill.style.background = "#546e7a";
    btnAuthAction.textContent = "Admin Login";
    loginBanner.style.display = "flex";
}

function logout() {
    setGuestState();
    loadModelVersions();
    loadAuditLogs();
}

btnAuthAction.addEventListener("click", () => {
    if (currentUser) {
        logout();
    } else {
        openLoginModal();
    }
});

if (btnBannerLogin) {
    btnBannerLogin.addEventListener("click", openLoginModal);
}

function openLoginModal() {
    loginError.style.display = "none";
    loginModal.style.display = "flex";
}

btnCancelLogin.addEventListener("click", () => {
    loginModal.style.display = "none";
});

loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.style.display = "none";

    try {
        const res = await fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: loginUsername.value.trim(),
                password: loginPassword.value
            })
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Invalid login credentials.");
        }

        authToken = data.access_token;
        localStorage.setItem("co2_auth_token", authToken);
        loginModal.style.display = "none";
        await checkAuthSession();
        loadModelVersions();
        loadAuditLogs();
    } catch (err) {
        loginError.textContent = err.message;
        loginError.style.display = "block";
    }
});

// -----------------------------------------------------------------------------
// 2. DATASET INGESTION & VALIDATION
// -----------------------------------------------------------------------------
datasetForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!datasetInput.files || datasetInput.files.length === 0) {
        showDatasetFeedback("Please select a CSV file.", false);
        return;
    }

    if (!authToken) {
        showDatasetFeedback("Please log in as an admin to upload datasets.", false);
        openLoginModal();
        return;
    }

    const formData = new FormData();
    formData.append("file", datasetInput.files[0]);

    showDatasetFeedback("Validating dataset schema and checking numeric bounds...", true);

    try {
        const res = await fetch("/api/admin/datasets", {
            method: "POST",
            headers: { "Authorization": `Bearer ${authToken}` },
            body: formData
        });
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.detail || "Dataset validation failed.");
        }
        showDatasetFeedback(`✅ Success! Validated ${data.row_count} rows across all 10 features. Saved as '${data.filename}'. Ready for training.`, true);
        datasetInput.value = "";
        loadAuditLogs();
    } catch (err) {
        showDatasetFeedback(`❌ Error: ${err.message}`, false);
    }
});

function showDatasetFeedback(msg, isSuccess) {
    datasetFeedback.textContent = msg;
    datasetFeedback.style.display = "block";
    datasetFeedback.style.background = isSuccess ? "#e8f5e9" : "#ffebee";
    datasetFeedback.style.color = isSuccess ? "#2e7d32" : "#c62828";
    datasetFeedback.style.border = `1px solid ${isSuccess ? "#a5d6a7" : "#ef9a9a"}`;
}

// -----------------------------------------------------------------------------
// 3. TRAINING MANAGEMENT & BACKGROUND POLLING
// -----------------------------------------------------------------------------
btnStartTraining.addEventListener("click", () => {
    if (!authToken) {
        openLoginModal();
        return;
    }

    showConfirmModal(
        "Start Model Retraining",
        "This will launch an asynchronous retraining job using 5-fold cross-validation across CatBoost, LightGBM, and XGBoost. The server will generate a candidate version upon completion.",
        async () => {
            try {
                btnStartTraining.disabled = true;
                trainingSpinner.style.display = "inline";
                trainingStatusTag.textContent = "Submitting...";

                const res = await fetch("/api/admin/train", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ dataset_name: "Processed_527_datapoints.csv" })
                });

                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || "Failed to trigger training.");
                }

                startJobPolling(data.job_id);
            } catch (err) {
                alert(`Training Error: ${err.message}`);
                trainingStatusTag.textContent = "Failed";
                trainingSpinner.style.display = "none";
                btnStartTraining.disabled = false;
            }
        }
    );
});

function startJobPolling(jobId) {
    trainingLogCard.style.display = "block";
    jobIdDisplay.textContent = jobId;
    jobStatusPill.textContent = "running";
    jobStatusPill.className = "badge-tag";
    jobDetailsDisplay.textContent = "Pipeline executing 5-fold CV across gradient-boosted trees...";
    trainingStatusTag.textContent = "Running";

    if (activePollingInterval) clearInterval(activePollingInterval);

    activePollingInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/admin/training/${jobId}`);
            if (!res.ok) return;
            const job = await res.json();

            jobStatusPill.textContent = job.status;
            if (job.status === "completed") {
                clearInterval(activePollingInterval);
                jobStatusPill.className = "badge-tag badge-best";
                jobDetailsDisplay.textContent = `Retraining finished successfully! Candidate version '${job.candidate_version}' created.`;
                if (job.metrics) {
                    jobMetricsDisplay.textContent = `Evaluation: R²=${job.metrics.R2?.toFixed(4)}, RMSE=${job.metrics.RMSE?.toFixed(4)}, MAE=${job.metrics.MAE?.toFixed(4)}`;
                }
                btnStartTraining.disabled = false;
                trainingSpinner.style.display = "none";
                trainingStatusTag.textContent = "Ready";
                loadModelVersions();
                loadAuditLogs();
            } else if (job.status === "failed") {
                clearInterval(activePollingInterval);
                jobStatusPill.className = "badge-tag badge-worst";
                jobDetailsDisplay.textContent = `Job failed: ${job.error_message || "Unknown error"}`;
                btnStartTraining.disabled = false;
                trainingSpinner.style.display = "none";
                trainingStatusTag.textContent = "Failed";
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }, 2000);
}

// -----------------------------------------------------------------------------
// 4. MODEL VERSIONS & ATOMIC ACTIVATION / ROLLBACK
// -----------------------------------------------------------------------------
async function loadModelVersions() {
    try {
        const res = await fetch("/api/admin/models");
        const data = await res.json();
        const models = data.models || [];

        adminModelsTbody.innerHTML = "";
        if (models.length === 0) {
            adminModelsTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 20px;">No models registered.</td></tr>`;
            return;
        }

        models.forEach(m => {
            const tr = document.createElement("tr");
            const isActive = m.status === "active";
            const isCandidate = m.status === "candidate";
            const isArchived = m.status === "archived";

            if (isActive) {
                heroActiveVersion.textContent = m.version;
                heroActiveR2.textContent = `Test R² Score: ${m.metrics?.R2?.toFixed(4) ?? "N/A"}`;
            }

            const statusClass = isActive ? "badge-best" : (isCandidate ? "badge-tag" : "");
            const statusLabel = isActive ? "ACTIVE" : (isCandidate ? "CANDIDATE" : "ARCHIVED");

            let actionButtons = "";
            if (isCandidate) {
                actionButtons = `<button type="button" class="primary" style="height: 28px; font-size: 11px; padding: 0 10px;" onclick="requestActivation('${m.version}')">Activate</button>`;
            } else if (isArchived) {
                actionButtons = `<button type="button" class="light" style="height: 28px; font-size: 11px; padding: 0 10px; border-color: #f57c00; color: #e65100;" onclick="requestRollback('${m.version}')">Rollback To This</button>`;
            } else {
                actionButtons = `<span style="font-size: 11.5px; color: var(--green); font-weight: 700;">● Active Model</span>`;
            }

            const weightsStr = m.weights ? Object.entries(m.weights).map(([k, v]) => `${k}: ${(v*100).toFixed(0)}%`).join(", ") : "N/A";

            tr.innerHTML = `
                <td><b>${m.version}</b></td>
                <td>${m.name || "Ensemble Model"}</td>
                <td><span class="badge-tag ${statusClass}" style="font-size: 11px;">${statusLabel}</span></td>
                <td><b>${m.metrics?.R2?.toFixed(4) ?? "-"}</b></td>
                <td>${m.metrics?.MAE?.toFixed(4) ?? "-"}</td>
                <td><small style="color: var(--muted);">${weightsStr}</small></td>
                <td>${actionButtons}</td>
            `;
            adminModelsTbody.appendChild(tr);
        });
    } catch (err) {
        adminModelsTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--danger); padding: 20px;">Failed to load models: ${err.message}</td></tr>`;
    }
}

window.requestActivation = function(version) {
    if (!authToken) {
        openLoginModal();
        return;
    }
    showConfirmModal(
        `Activate Version ${version}?`,
        `This will atomically switch production traffic to model version '${version}'. The currently active model will be archived.`,
        async () => {
            try {
                const res = await fetch(`/api/admin/models/${version}/activate`, {
                    method: "POST",
                    headers: { "Authorization": `Bearer ${authToken}` }
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Activation failed.");
                alert(`Success: ${data.message}`);
                loadModelVersions();
                loadAuditLogs();
            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        }
    );
};

window.requestRollback = function(version) {
    if (!authToken) {
        openLoginModal();
        return;
    }
    showConfirmModal(
        `Rollback to Version ${version}?`,
        `Are you sure you want to revert production back to version '${version}'? This action is tracked in the system audit log.`,
        async () => {
            try {
                const res = await fetch(`/api/admin/models/${version}/rollback`, {
                    method: "POST",
                    headers: { "Authorization": `Bearer ${authToken}` }
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Rollback failed.");
                alert(`Success: ${data.message}`);
                loadModelVersions();
                loadAuditLogs();
            } catch (err) {
                alert(`Error: ${err.message}`);
            }
        }
    );
};

btnRefreshModels.addEventListener("click", loadModelVersions);

// -----------------------------------------------------------------------------
// 5. AUDIT LOGS
// -----------------------------------------------------------------------------
async function loadAuditLogs() {
    try {
        const res = await fetch("/api/admin/audit-logs");
        const data = await res.json();
        const logs = data.audit_logs || [];

        auditLogsTbody.innerHTML = "";
        if (logs.length === 0) {
            auditLogsTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; padding: 15px; color: var(--muted);">No administrative actions recorded yet.</td></tr>`;
            return;
        }

        logs.forEach(l => {
            const tr = document.createElement("tr");
            const dt = l.timestamp ? new Date(l.timestamp).toLocaleString() : "-";
            tr.innerHTML = `
                <td><small style="color: var(--muted);">${dt}</small></td>
                <td><b>${l.username || "system"}</b></td>
                <td><span class="badge-tag" style="font-size: 11px;">${l.action}</span></td>
                <td>${l.details || "-"}</td>
            `;
            auditLogsTbody.appendChild(tr);
        });
    } catch (err) {
        auditLogsTbody.innerHTML = `<tr><td colspan="4" style="text-align: center; color: var(--danger); padding: 15px;">Failed to load audit logs.</td></tr>`;
    }
}

// -----------------------------------------------------------------------------
// 6. CONFIRMATION MODAL HELPER
// -----------------------------------------------------------------------------
function showConfirmModal(title, msg, onConfirm) {
    confirmTitle.textContent = title;
    confirmMessage.textContent = msg;
    pendingConfirmAction = onConfirm;
    confirmModal.style.display = "flex";
}

btnConfirmCancel.addEventListener("click", () => {
    confirmModal.style.display = "none";
    pendingConfirmAction = null;
});

btnConfirmOk.addEventListener("click", async () => {
    confirmModal.style.display = "none";
    if (pendingConfirmAction) {
        const act = pendingConfirmAction;
        pendingConfirmAction = null;
        await act();
    }
});

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    checkAuthSession();
    loadModelVersions();
    loadAuditLogs();
});
