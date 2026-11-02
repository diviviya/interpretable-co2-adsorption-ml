# Backend and Admin Workflow Setup

## 1. Install dependencies

From the repository root, create and activate a virtual environment, then install the single root dependency file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Use the pinned package versions in `requirements.txt` when reproducing the saved ML artifacts. In particular, do not casually change scikit-learn, CatBoost, or XGBoost versions because serialized estimators can be version-sensitive.

## 2. Configure local secrets

Copy `backend/.env.example` to `backend/.env` and replace the placeholders:

```env
ENVIRONMENT=development
JWT_SECRET=generate-a-random-secret-at-least-32-characters-long
JWT_EXPIRATION_HOURS=24
ADMIN_USERNAME=projectadmin
ADMIN_PASSWORD=use-a-strong-password
```

The `.env` file is ignored by Git. Never commit real credentials. On first startup, the backend creates the administrator specified by `ADMIN_USERNAME` and `ADMIN_PASSWORD` if that account does not already exist. Public registration always creates a normal `user` account.

## 3. Start the application

From the repository root:

```bash
python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the application, `http://127.0.0.1:8000/login.html` to sign in, and `http://127.0.0.1:8000/admin.html` for the administrator console. FastAPI's interactive API documentation is at `http://127.0.0.1:8000/docs`.

## 4. Admin retraining workflow

1. Sign in with the administrator account configured in `.env`.
2. Open Admin.
3. Upload a labelled CSV containing the ten feature columns and `CO2 uptake (mmol/g)`.
4. Select the uploaded dataset and start training.
5. Wait for the background job to complete and inspect the validation metrics.
6. Review the candidate model version. Activate only after review.

Admin retraining creates a fresh 80/20 train/validation split. It does not use or overwrite the original frozen research test split or the v1.0 deployment artifact. The candidate's metrics are validation metrics, not independent test metrics. A dataset made from a subset of the original observations is only a technical demonstration, not new experimental evidence.

## 5. Architecture in one sentence

The browser sends HTTP requests to FastAPI routes; routes validate and delegate work to service modules; services call the ML pipeline or JSON storage; JSON responses return to the browser for display.

## 6. Tests

```bash
python3 -m pytest -q
```

The saved model artifacts were created with the versions pinned in the root requirements file. Warnings or failures under different versions should be investigated in the pinned environment before changing model artifacts.

## Current implementation limits

- Storage is JSON-file based, suitable for a small academic demonstration, not concurrent production workloads.
- Background training uses an in-process thread. Restarting the server interrupts an active job; a production deployment should use a durable job queue.
- Admin candidate models are stored as joblib bundles and must be loaded with compatible Python/ML package versions.
- Deployment still requires configuring the host's environment variables and verifying the deployed application end to end.
