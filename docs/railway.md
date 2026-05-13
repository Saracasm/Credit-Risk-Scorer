# Deploy on Railway (API + web)

Two services from this repo: **FastAPI** (Python) and **Next.js** (Node). No Docker required.

## 1. Train and provide the model

Locally (or in CI), run `python src/train.py` so `models/xgb_pipeline.pkl` exists.

Railway does not get `.pkl` from git if it is gitignored. Pick one:

- **Volume:** Attach a Railway volume, upload the file once, set `MODEL_PATH` to that path.
- **Artifact URL:** Store the pickle in cloud storage and download it in a startup script (not included here).
- **Commit for demo only:** Remove `*.pkl` from `.gitignore` temporarily (not recommended for large files).

## 2. Service A — API

- **Root directory:** repository root (or leave default).
- **Install:** `pip install -r backend/requirements.txt`
- **Start:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

**Variables**

| Name | Example |
|------|---------|
| `MODEL_PATH` | `/data/xgb_pipeline.pkl` if using a volume |
| `CORS_ORIGINS` | `https://your-frontend.up.railway.app` (comma-separated if several) |

After deploy, note the public URL (e.g. `https://credit-api.up.railway.app`).

## 3. Service B — Frontend

- **Root directory:** `frontend`
- **Install:** `npm install`
- **Build:** `npm run build`
- **Start:** `npm start`

**Variables**

| Name | Example |
|------|---------|
| `NEXT_PUBLIC_API_URL` | `https://credit-api.up.railway.app` (no trailing slash) |

Redeploy the frontend after changing `NEXT_PUBLIC_API_URL` (it is baked in at build time).

## 4. Local dev

From repo root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Other terminal:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. The UI calls `http://127.0.0.1:8000` unless `.env.local` overrides `NEXT_PUBLIC_API_URL`.
