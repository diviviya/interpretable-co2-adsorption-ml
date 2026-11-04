// =====================================================================
// api.js — every call to the backend goes through this file.
// Pages never call fetch() themselves; they call API.something().
// In mock mode each function hands the job to MOCK (mock.js) instead.
//
// Logged-in requests carry the session token automatically
// ("Authorization: Bearer <token>"), so pages don't deal with it.
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

  // Requests where a 401 means "wrong password", not "session expired".
  const AUTH_FORMS = ["/api/auth/login", "/api/auth/signup"];

  // One shared helper for every request: token, timeout, JSON, readable errors.
  async function request(path, { method = "GET", body, form } = {}) {
    const headers = {};
    let payload;
    if (form) {
      payload = form; // FormData: the browser sets the right Content-Type itself
    } else if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    const token = Auth.token();
    if (token) headers["Authorization"] = "Bearer " + token;

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
      const message = readableDetail(data && (data.detail || data.message)) || `Server error (${res.status}).`;
      // Session missing or expired on a protected page: back to the login page.
      if (res.status === 401 && !AUTH_FORMS.includes(path)) {
        Auth.clear();
        const need = document.body.dataset.protected;
        if (need) location.replace(loginPageFor(need));
      }
      const error = new Error(message);
      error.status = res.status;
      throw error;
    }
    return data;
  }

  const m = () => CONFIG.USE_MOCK;

  return {
    // ---- accounts ----
    signup: (username, password) =>
      m() ? MOCK.signup(username, password) : request("/api/auth/signup", { method: "POST", body: { username, password } }),
    login: (username, password, admin = false) =>
      m() ? MOCK.login(username, password, admin) : request("/api/auth/login", { method: "POST", body: { username, password, admin } }),
    me:     () => m() ? MOCK.me()     : request("/api/auth/me"),
    logout: () => m() ? MOCK.logout() : request("/api/auth/logout", { method: "POST" }),

    // ---- public model info ----
    health:          () => m() ? MOCK.health()          : request("/api/health"),
    getFeatureInfo:  () => m() ? MOCK.getFeatureInfo()  : request("/api/features"),
    getActiveModel:  () => m() ? MOCK.getActiveModel()  : request("/api/models/active"),
    getImportance:   () => m() ? MOCK.getImportance()   : request("/api/models/active/importance"),

    // ---- logged-in users ----
    getHistory: () => m() ? MOCK.getHistory() : request("/api/predictions/history"),
    predict: (features) =>
      m() ? MOCK.predict(features) : request("/api/predict", { method: "POST", body: { features } }),
    predictBatch: (file) => {
      if (m()) return MOCK.predictBatch(file);
      const form = new FormData();
      form.append("file", file);
      return request("/api/predict/batch", { method: "POST", form });
    },

    // ---- admins (the backend checks the role on every call) ----
    uploadDataset: (file) => {
      if (m()) return MOCK.uploadDataset(file);
      const form = new FormData();
      form.append("file", file);
      return request("/api/admin/datasets", { method: "POST", form });
    },
    startTraining: (datasetId) =>
      m() ? MOCK.startTraining(datasetId)
          : request("/api/admin/train", { method: "POST", body: { dataset_id: datasetId } }),
    getTrainingStatus: (jobId) =>
      m() ? MOCK.getTrainingStatus(jobId)
          : request(`/api/admin/training/${encodeURIComponent(jobId)}`),
    listModels: () => m() ? MOCK.listModels() : request("/api/admin/models"),
    activateModel: (version) =>
      m() ? MOCK.activateModel(version)
          : request(`/api/admin/models/${encodeURIComponent(version)}/activate`, { method: "POST" }),
    rollbackModel: (version) =>
      m() ? MOCK.rollbackModel(version)
          : request(`/api/admin/models/${encodeURIComponent(version)}/rollback`, { method: "POST" }),
  };
})();
