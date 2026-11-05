# CarbonLab: frontend

Plain HTML, CSS and JavaScript. No build step, no frameworks. The frontend does no machine learning: it sends inputs to the FastAPI backend and shows what comes back.

## Run it

The FastAPI backend serves these pages itself. From the repo root:

```
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/ (or share it with a Cloudflare Tunnel link).
For demo mode without the backend: `python -m http.server 5500` and open http://localhost:5500/frontend/.

Don't open the HTML files by double-clicking them. Demo mode reads `artifacts/feature_info.json` and `results/08_shap_importance.csv`, and browsers block that for pages opened as files.

## Demo mode vs real backend

`js/config.js`:

- `USE_MOCK: true` uses fake responses from `js/mock.js`. Every page shows a "Demo mode" banner.
- `USE_MOCK: false` sends real requests to `API_BASE` (default `http://127.0.0.1:8000`).

Demo mode is for building the frontend only. Turn it off before the final demo.

In demo mode the **feature ranges, global SHAP importance and the active model's test metrics are real** (read from the project files and Notebook 07). **Individual predictions, their SHAP values, training jobs and candidate metrics are fake.**

## Files

| File | Purpose |
|---|---|
| `index.html` | Public landing page: what the tool does, how the model was built, live model scores |
| `login.html` + `js/login.js` | Log in / create a user account |
| `admin-login.html` + `js/login.js` | Admin login (admin accounts only) |
| `predict.html` + `js/predict.js` | Prediction form, result, per-prediction SHAP chart |
| `batch.html` + `js/batch.js` | CSV upload, results table, CSV download |
| `history.html` + `js/history.js` | Past predictions (from the backend) |
| `about.html` + `js/about.js` | Purpose, method, live metrics, global importance, limitations |
| `admin.html` + `js/admin.js` | Dataset upload, training, model versions (admins only) |
| `js/config.js` | Settings and the list of the 10 inputs |
| `js/api.js` | Every backend call (pages never call `fetch` directly) |
| `js/mock.js` | Fake backend for demo mode |
| `js/common.js` | Login state, page guard, navigation, lattice illustration, CSV and formatting helpers |
| `css/style.css` | All styles |

## API contract (proposed: confirm with the backend teammate)

Feature names are always the **exact dataset column names**, e.g. `"Surface Area (m2/g)"`. An optional input left blank is sent as `null`; the backend fills it with the training median.

### `GET /api/health`
```json
{ "status": "ok" }
```

### `GET /api/features`  *(added by the frontend: serves feature_info.json)*
```json
{ "Surface Area (m2/g)": { "min": 2.48, "max": 3337.0, "median": 1420.0 }, "...": {} }
```

### `POST /api/predict`
Request:
```json
{ "features": {
    "Surface Area (m2/g)": 1500, "Total Pore Volume(cm3/g)": 0.8, "Micropore Volume (cm3/g)": null,
    "C (%)": 80, "H (%)": 2, "N (%)": 3, "O (%)": 12, "S (%)": 0.1,
    "Temp (°C)": 25, "Pressure (bar)": 1 } }
```
Response:
```json
{
  "prediction": 2.91,
  "unit": "mmol/g",
  "model_version": "v1-ensemble",
  "base_value": 2.6,
  "shap": [ { "feature": "Pressure (bar)", "value": 1, "contribution": 0.42 } ],
  "explanation": null,
  "warnings": []
}
```
`base_value` + the sum of `contribution` should equal `prediction` (SHAP values add up). `shap` has one entry per input. `explanation` is optional text; if it's `null` the page writes a summary from the SHAP numbers.

### `POST /api/predict/batch`
Multipart form, field `file` (CSV). Response:
```json
{ "model_version": "v1-ensemble",
  "rows": [ { "row": 2, "Surface Area (m2/g)": 1500, "...": 0, "prediction": 2.91 } ],
  "errors": [ { "row": 6, "message": "\"Surface Area (m2/g)\" is not a number" } ] }
```
`row` is the line number in the CSV (header = line 1).

### `GET /api/models/active`
```json
{ "version": "v1-ensemble", "created_at": "2026-10-28T15:15:00+05:30",
  "model_type": "Entropy-TOPSIS weighted ensemble of 6 tree models", "status": "active",
  "dataset": "Yuan et al. (2021) biomass carbons, 526 rows",
  "metrics": { "rmse": 0.644, "mae": 0.477, "r2": 0.848 }, "metrics_split": "test (106 rows)" }
```

### `GET /api/models/active/importance`  *(added by the frontend)*
```json
{ "model_version": "v1-ensemble",
  "importance": [ { "feature": "Temp (°C)", "mean_abs_shap": 0.639 } ] }
```

### `GET /api/predictions/history`  *(added by the frontend)*
```json
{ "items": [ { "timestamp": "2026-10-28T16:30:00+05:30", "model_version": "v1-ensemble",
               "prediction": 2.91, "features": { "Surface Area (m2/g)": 1500 } } ] }
```

### Accounts
Logged-in requests send `Authorization: Bearer <session token>` (added automatically by `api.js`).
Pages with `<body data-protected="user">` or `"admin"` send visitors without a valid session to the login page.

| Endpoint | Request | Response |
|---|---|---|
| `POST /api/auth/signup` | `{ username, password }` | `{ token, user: {id, username, role} }` (creates a normal user) |
| `POST /api/auth/login` | `{ username, password, admin? }` | `{ token, user }`; `401` wrong details, `403` if `admin: true` and the account isn't an admin |
| `GET /api/auth/me` | | `{ user }`, or `401` if the session ended |
| `POST /api/auth/logout` | | `{ ok: true }` |

Predict, batch and history need a logged-in user (history shows only your own predictions).
The admin account comes from `ADMIN_USERNAME` / `ADMIN_PASSWORD` in `backend/.env`.

### Admin endpoints
All need an admin session. The backend checks the role on every call and replies `401`/`403` otherwise.

| Endpoint | Request | Response |
|---|---|---|
| `POST /api/admin/datasets` | multipart `file` | `{ dataset_id, filename, rows, columns: [...], missing_values: {col: n}, duplicate_rows, errors: [...] }` |
| `POST /api/admin/train` | `{ "dataset_id": "..." }` | `{ job_id }` |
| `GET /api/admin/training/{job_id}` | | `{ job_id, status, progress: 0–1, logs: [...], candidate?, error? }` |
| `GET /api/admin/models` | | `{ models: [...] }` (`status` = `active`/`candidate`/`previous`) |
| `POST /api/admin/models/{version}/activate` | | `{ models: [...] }` |
| `POST /api/admin/models/{version}/rollback` | | `{ models: [...] }` |
| `GET /api/admin/users` | | `{ users: [{id, username, role, created_at, predictions, main_admin}] }` |
| `POST /api/admin/users/{id}/role` | `{ "role": "admin"\|"user" }` | `{ users: [...] }`; `409` for your own role, the main admin, or the last admin |

Errors from any endpoint: a non-2xx status with FastAPI's usual `{ "detail": ... }` body. The page shows the detail text.

