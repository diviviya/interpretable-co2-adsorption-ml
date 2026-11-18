// =====================================================================
// api.js — every call to the backend goes through this file.
// Pages never call fetch() themselves; they call API.something().
// In mock mode each function hands the job to MOCK (mock.js) instead.
// =====================================================================

const API = (() => {
  // Turn FastAPI's error "detail" (a string or a list) into one readable line.
  function readableDetail(detail) {
    if (!detail) return "";
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => {
          const where = Array.isArray(d.loc) ? d.loc.filter((p) => p !== "body").join(" → ") : "";
          return where ? `${where}: ${d.msg}` : d.msg;
        })
        .join("; ");
    }
    return String(detail);
  }

  let guestTokenPromise = null;
  async function getSessionToken() {
    try {
      let t = sessionStorage.getItem("app_token");
      if (t) return t;
      if (!guestTokenPromise) {
        guestTokenPromise = fetch(CONFIG.API_BASE + "/api/auth/guest", { method: "POST" })
          .then((r) => (r.ok ? r.json() : null))
          .then((d) => {
            if (d && d.access_token) {
              sessionStorage.setItem("app_token", d.access_token);
              return d.access_token;
            }
            return null;
          })
          .catch(() => null)
          .finally(() => { guestTokenPromise = null; });
      }
      return await guestTokenPromise;
    } catch {
      return null;
    }
  }

  // One shared helper for every request: timeout, JSON, readable errors.
  async function request(path, { method = "GET", body, form, adminKey } = {}) {
    const headers = {};
    let payload;
    if (form) {
      payload = form; // FormData: the browser sets the right Content-Type itself
    } else if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    if (adminKey) {
      headers["Authorization"] = "Bearer " + adminKey;
    } else {
      const sessionToken = await getSessionToken();
      if (sessionToken) headers["Authorization"] = "Bearer " + sessionToken;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
    let res;
    try {
      res = await fetch(CONFIG.API_BASE + path, { method, headers, body: payload, signal: controller.signal });
    } catch (err) {
      throw new Error(err.name === "AbortError"
        ? "The server took too long to respond."
        : "Couldn't reach the server. Is the backend running?");
    } finally {
      clearTimeout(timer);
    }

    let data = null;
    try { data = await res.json(); } catch { /* empty or non-JSON reply */ }

    if (!res.ok) {
      if (res.status === 401 || res.status === 403) {
        if (authErrorHandler && !path.endsWith("/login")) authErrorHandler();
        throw new Error("Not authorised. Check the admin key.");
      }
      throw new Error(readableDetail(data && (data.detail || data.message)) || `Server error (${res.status}).`);
    }
    return data;
  }

  const m = () => CONFIG.USE_MOCK;

  // The admin page registers a function to run when the server rejects the key.
  let authErrorHandler = null;

  return {
    onAuthError: (fn) => { authErrorHandler = fn; },

    // ---- user side ----
    health:          ()         => m() ? MOCK.health()          : request("/api/health"),
    getFeatureInfo:  ()         => m() ? MOCK.getFeatureInfo()  : request("/api/features"),
    getActiveModel:  ()         => m() ? MOCK.getActiveModel()  : request("/api/models/active"),
    getImportance:   ()         => m() ? MOCK.getImportance()   : request("/api/models/active/importance"),
    getHistory:      ()         => m() ? MOCK.getHistory()      : request("/api/predictions/history"),

    predict: (features) =>
      m() ? MOCK.predict(features) : request("/api/predict", { method: "POST", body: { features } }),

    getSelectionGatePrecursors: () =>
      m() ? MOCK.getSelectionGatePrecursors() : request("/api/selection-gate/precursors"),

    runSelectionGate: (payload) =>
      m() ? MOCK.runSelectionGate(payload) : request("/api/selection-gate", { method: "POST", body: payload }),

    predictBatch: (file) => {
      if (m()) return MOCK.predictBatch(file);
      const form = new FormData();
      form.append("file", file);
      return request("/api/predict/batch", { method: "POST", form });
    },

    // ---- admin side (backend must check the key on every call) ----
    adminLogin: (adminKey) =>
      m() ? MOCK.adminLogin(adminKey) : request("/api/admin/login", { method: "POST", adminKey }),
    uploadDataset: (file, adminKey) => {
      if (m()) return MOCK.uploadDataset(file);
      const form = new FormData();
      form.append("file", file);
      return request("/api/admin/datasets", { method: "POST", form, adminKey });
    },
    startTraining: (datasetId, adminKey) =>
      m() ? MOCK.startTraining(datasetId)
          : request("/api/admin/train", { method: "POST", body: { dataset_id: datasetId }, adminKey }),
    getTrainingStatus: (jobId, adminKey) =>
      m() ? MOCK.getTrainingStatus(jobId)
          : request(`/api/admin/training/${encodeURIComponent(jobId)}`, { adminKey }),
    listModels: (adminKey) =>
      m() ? MOCK.listModels() : request("/api/admin/models", { adminKey }),
    activateModel: (version, adminKey) =>
      m() ? MOCK.activateModel(version)
          : request(`/api/admin/models/${encodeURIComponent(version)}/activate`, { method: "POST", adminKey }),
    rollbackModel: (version, adminKey) =>
      m() ? MOCK.rollbackModel(version)
          : request(`/api/admin/models/${encodeURIComponent(version)}/rollback`, { method: "POST", adminKey }),
  };
})();
