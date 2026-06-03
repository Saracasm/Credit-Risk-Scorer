"""
Credit Risk Scorer — FastAPI Backend.

Run from repository root:
  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health            – model status
  POST /api/predict       – score one applicant (SHAP + risk)
  POST /api/scenarios     – compare up to 3 what-if scenarios
  POST /api/batch         – batch-score a JSON array of applicants
  GET  /api/dashboard     – portfolio stats + recent predictions
  POST /api/advisor       – one-shot AI advisor chat message
  POST /api/report        – generate PDF report (returns binary)

Env vars:
  MODEL_PATH   – path to xgb_pipeline.pkl (default: models/xgb_pipeline.pkl)
  CORS_ORIGINS – comma-separated (default: http://localhost:3000)
  GEMINI_API_KEY – for free-tier AI advisor
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np

# Repo root must be on path for `import src.*`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.inference import (
    build_feature_row,
    load_pipeline,
    predict_default,
    shap_values_row,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_pipe: object | None = None
_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe, _load_error
    _load_error = None
    _pipe = None
    try:
        mp = os.environ.get("MODEL_PATH", "").strip()
        _pipe = load_pipeline(Path(mp) if mp else None)
    except Exception as e:
        _load_error = str(e)
    yield
    _pipe = None


app = FastAPI(
    title="Credit Risk Scorer API",
    description="Default probability + SHAP for the Give Me Some Credit model.",
    version="2.0.0",
    lifespan=lifespan,
)

_cors = os.environ.get("CORS_ORIGINS", "http://localhost:3000").strip()
_origins = [o.strip() for o in _cors.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_pipe():
    if _pipe is None:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model not loaded. Train with python src/train.py.",
        )
    return _pipe


def _risk_label(p: float) -> str:
    if p > 0.6:
        return "HIGH"
    if p > 0.3:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ApplicantInput(BaseModel):
    """Raw applicant inputs (same semantics as Kaggle columns)."""
    age: float = Field(ge=18, le=120)
    monthly_income: float = Field(ge=0, le=500_000)
    debt_ratio: float = Field(ge=0, le=10)
    revolving_util: float = Field(ge=0, le=1)
    n_30_59_late: float = Field(ge=0, le=20)
    n_60_89_late: float = Field(ge=0, le=20)
    n_90_late: float = Field(ge=0, le=20)
    n_open_lines: float = Field(ge=0, le=50)
    n_real_estate: float = Field(ge=0, le=20)
    n_dependents: float = Field(ge=0, le=15)


class ShapItem(BaseModel):
    feature: str
    value: float


class PredictResponse(BaseModel):
    probability: float
    predicted_class: int
    risk_label: str
    shap: list[ShapItem]
    shap_base_value: float


class ScenarioRequest(BaseModel):
    baseline: ApplicantInput
    scenarios: list[ApplicantInput]  # up to 3


class ScenarioResult(BaseModel):
    probability: float
    risk_label: str
    change: float  # relative to baseline
    shap: list[ShapItem]


class ScenarioResponse(BaseModel):
    baseline_probability: float
    baseline_risk_label: str
    results: list[ScenarioResult]


class BatchItem(BaseModel):
    """One row in a batch request."""
    age: float
    monthly_income: float = Field(alias="MonthlyIncome", default=0)
    debt_ratio: float = Field(alias="DebtRatio", default=0)
    revolving_util: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines", default=0
    )
    n_30_59_late: float = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse", default=0
    )
    n_60_89_late: float = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse", default=0
    )
    n_90_late: float = Field(alias="NumberOfTimes90DaysLate", default=0)
    n_open_lines: float = Field(
        alias="NumberOfOpenCreditLinesAndLoans", default=0
    )
    n_real_estate: float = Field(alias="NumberRealEstateLoansOrLines", default=0)
    n_dependents: float = Field(alias="NumberOfDependents", default=0)

    model_config = {"populate_by_name": True}


class BatchRequest(BaseModel):
    applicants: list[dict]


class BatchResultRow(BaseModel):
    index: int
    probability: float
    predicted_class: int
    risk_label: str


class BatchResponse(BaseModel):
    total: int
    avg_risk: float
    high_risk_count: int
    low_risk_count: int
    rows: list[BatchResultRow]


class DashboardResponse(BaseModel):
    total: int
    avg_prob: float
    high_count: int
    medium_count: int
    low_count: int
    recent: list[dict]
    histogram: list[float]  # raw probabilities for client-side histogram


class AdvisorRequest(BaseModel):
    message: str
    applicant: ApplicantInput | None = None
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    api_key: str | None = None  # pro tier; free tier uses env
    session_id: str | None = None  # carries conversation memory across turns


class AdvisorResponse(BaseModel):
    reply: str
    session_id: str


class ReportRequest(BaseModel):
    applicant: ApplicantInput


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    if _load_error:
        return {"ok": False, "model_loaded": False, "error": _load_error}
    return {"ok": True, "model_loaded": _pipe is not None}


@app.get("/api/knowledge")
def knowledge():
    """Status of the policy RAG knowledge base (for the advisor UI badge)."""
    try:
        from src.knowledge import knowledge_info
        return knowledge_info()
    except Exception:
        return {"available": False, "doc_count": 0, "chunk_count": 0, "docs": []}


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.post("/api/predict", response_model=PredictResponse)
def predict(body: ApplicantInput):
    """Score one applicant; returns probability and SHAP contributions."""
    pipe = _require_pipe()
    X = build_feature_row(
        age=body.age, monthly_income=body.monthly_income,
        debt_ratio=body.debt_ratio, revolving_util=body.revolving_util,
        n_30_59_late=body.n_30_59_late, n_60_89_late=body.n_60_89_late,
        n_90_late=body.n_90_late, n_open_lines=body.n_open_lines,
        n_real_estate=body.n_real_estate, n_dependents=body.n_dependents,
    )
    prob, cls = predict_default(pipe, X)
    names, svals, base = shap_values_row(pipe, X)
    order = sorted(range(len(svals)), key=lambda i: abs(svals[i]), reverse=True)
    shap_out = [ShapItem(feature=names[i], value=float(svals[i])) for i in order[:16]]

    # Persist to SQLite
    try:
        from src.database import save_prediction
        save_prediction(body.model_dump(), prob, cls)
    except Exception:
        pass

    return PredictResponse(
        probability=round(prob, 6),
        predicted_class=cls,
        risk_label=_risk_label(prob),
        shap=shap_out,
        shap_base_value=round(base, 6),
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@app.post("/api/scenarios", response_model=ScenarioResponse)
def scenarios(body: ScenarioRequest):
    """Compare baseline vs up to 3 modified scenarios."""
    pipe = _require_pipe()
    b = body.baseline
    X_base = build_feature_row(
        age=b.age, monthly_income=b.monthly_income,
        debt_ratio=b.debt_ratio, revolving_util=b.revolving_util,
        n_30_59_late=b.n_30_59_late, n_60_89_late=b.n_60_89_late,
        n_90_late=b.n_90_late, n_open_lines=b.n_open_lines,
        n_real_estate=b.n_real_estate, n_dependents=b.n_dependents,
    )
    prob_base, _ = predict_default(pipe, X_base)

    results = []
    for sc in body.scenarios[:3]:
        X_sc = build_feature_row(
            age=sc.age, monthly_income=sc.monthly_income,
            debt_ratio=sc.debt_ratio, revolving_util=sc.revolving_util,
            n_30_59_late=sc.n_30_59_late, n_60_89_late=sc.n_60_89_late,
            n_90_late=sc.n_90_late, n_open_lines=sc.n_open_lines,
            n_real_estate=sc.n_real_estate, n_dependents=sc.n_dependents,
        )
        prob_sc, _ = predict_default(pipe, X_sc)
        names, svals, _ = shap_values_row(pipe, X_sc)
        order = sorted(range(len(svals)), key=lambda i: abs(svals[i]), reverse=True)
        shap_out = [ShapItem(feature=names[i], value=float(svals[i])) for i in order[:8]]
        results.append(ScenarioResult(
            probability=round(prob_sc, 6),
            risk_label=_risk_label(prob_sc),
            change=round(prob_sc - prob_base, 6),
            shap=shap_out,
        ))

    return ScenarioResponse(
        baseline_probability=round(prob_base, 6),
        baseline_risk_label=_risk_label(prob_base),
        results=results,
    )


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

@app.post("/api/batch", response_model=BatchResponse)
def batch_score(body: BatchRequest):
    """Score a batch of applicants provided as a JSON array."""
    pipe = _require_pipe()
    import pandas as pd
    from src.feature_engineering import add_engineered_features

    df = pd.DataFrame(body.applicants)
    required = [
        "age", "MonthlyIncome", "DebtRatio",
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "NumberOfDependents",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    df_eng = add_engineered_features(df)
    probs = pipe.predict_proba(df_eng)[:, 1]

    rows = []
    for i, p in enumerate(probs):
        rows.append(BatchResultRow(
            index=i,
            probability=round(float(p), 6),
            predicted_class=int(p >= 0.5),
            risk_label=_risk_label(float(p)),
        ))

    return BatchResponse(
        total=len(probs),
        avg_risk=round(float(probs.mean()), 4),
        high_risk_count=int((probs > 0.6).sum()),
        low_risk_count=int((probs < 0.3).sum()),
        rows=rows,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/api/dashboard", response_model=DashboardResponse)
def dashboard():
    """Return portfolio stats and recent predictions."""
    try:
        from src.database import get_all_predictions, get_prediction_stats
        preds = get_all_predictions(limit=1000)
        stats = get_prediction_stats()
    except Exception:
        return DashboardResponse(
            total=0, avg_prob=0, high_count=0, medium_count=0,
            low_count=0, recent=[], histogram=[],
        )

    if preds.empty or stats.get("total", 0) == 0:
        return DashboardResponse(
            total=0, avg_prob=0, high_count=0, medium_count=0,
            low_count=0, recent=[], histogram=[],
        )

    display_cols = [c for c in ["timestamp", "age", "monthly_income", "probability", "risk_level"]
                    if c in preds.columns]
    recent = preds[display_cols].head(20).to_dict("records")
    histogram = preds["probability"].dropna().tolist()

    return DashboardResponse(
        total=stats.get("total", 0),
        avg_prob=round(stats.get("avg_prob", 0) or 0, 4),
        high_count=stats.get("high_count", 0) or 0,
        medium_count=stats.get("med_count", 0) or 0,
        low_count=stats.get("low_count", 0) or 0,
        recent=recent,
        histogram=histogram,
    )


# ---------------------------------------------------------------------------
# AI Advisor (stateless per-message)
# ---------------------------------------------------------------------------

@app.post("/api/advisor", response_model=AdvisorResponse)
def advisor(body: AdvisorRequest):
    """Send a message to the AI credit advisor. Uses Gemini or OpenAI.

    Includes automatic retry with exponential backoff for rate-limit (429)
    errors so free-tier Gemini usage stays smooth.
    """
    api_key = body.api_key
    if not api_key:
        if body.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
        elif body.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
        elif body.provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

    if not api_key:
        raise HTTPException(400, f"No API key provided for {body.provider}. Please set {body.provider.upper()}_API_KEY in backend .env or provide it in the request.")

    from src.agent import create_chat_session
    applicant_dict = body.applicant.model_dump() if body.applicant else None

    # Remap keys to match what agent expects
    if applicant_dict:
        applicant_dict = {
            "age": applicant_dict["age"],
            "monthly_income": applicant_dict["monthly_income"],
            "debt_ratio": applicant_dict["debt_ratio"],
            "revolving_util": applicant_dict["revolving_util"],
            "n_30_59_late": applicant_dict["n_30_59_late"],
            "n_60_89_late": applicant_dict["n_60_89_late"],
            "n_90_late": applicant_dict["n_90_late"],
            "n_open_lines": applicant_dict["n_open_lines"],
            "n_real_estate": applicant_dict["n_real_estate"],
            "n_dependents": applicant_dict["n_dependents"],
        }

    # --- Conversation memory -------------------------------------------------
    # Reuse the caller's session_id (or mint a new one) and replay prior turns so
    # follow-up questions like "tell me more about that second factor" work.
    session_id = body.session_id or str(uuid.uuid4())
    history: list[dict] = []
    try:
        from src.database import get_conversation
        history = get_conversation(session_id)[-20:]  # cap to recent turns
    except Exception:
        history = []

    MAX_RETRIES = 3
    BASE_DELAY = 2  # seconds; delays will be 2, 4, 8

    for attempt in range(MAX_RETRIES + 1):
        try:
            session = create_chat_session(
                provider=body.provider,
                api_key=api_key,
                model_name=body.model_name,
                applicant=applicant_dict,
                history=history,
            )
            response = session.send_message(body.message)
            reply = response.text
            break  # success
        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "429" in err
                or "quota" in err.lower()
                or "rate" in err.lower()
                or "resource" in err.lower()
            )
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = BASE_DELAY * (2 ** attempt)  # 2s, 4s, 8s
                time.sleep(wait)
                continue
            # Final attempt or non-rate-limit error
            if is_rate_limit:
                reply = "⚠️ Rate limit reached after retries. Please wait ~60s and try again."
            else:
                reply = f"❌ Error: {err[:300]}"
            break

    # Persist this turn so the next request can recall it.
    try:
        from src.database import save_message
        save_message(session_id, "user", body.message)
        save_message(session_id, "assistant", reply)
    except Exception:
        pass

    return AdvisorResponse(reply=reply, session_id=session_id)


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------

@app.post("/api/report")
def report(body: ReportRequest):
    """Generate a PDF credit assessment report."""
    pipe = _require_pipe()
    a = body.applicant
    X = build_feature_row(
        age=a.age, monthly_income=a.monthly_income,
        debt_ratio=a.debt_ratio, revolving_util=a.revolving_util,
        n_30_59_late=a.n_30_59_late, n_60_89_late=a.n_60_89_late,
        n_90_late=a.n_90_late, n_open_lines=a.n_open_lines,
        n_real_estate=a.n_real_estate, n_dependents=a.n_dependents,
    )
    prob, cls = predict_default(pipe, X)
    names, svals, _ = shap_values_row(pipe, X)

    try:
        from src.agent import get_loan_recommendation
        loan_rec = get_loan_recommendation(
            a.age, a.monthly_income, a.debt_ratio, a.revolving_util,
            int(a.n_30_59_late), int(a.n_60_89_late), int(a.n_90_late),
            int(a.n_open_lines), int(a.n_real_estate), int(a.n_dependents),
        )
    except Exception:
        loan_rec = None

    from src.report_generator import generate_report
    pdf_bytes = generate_report(
        a.model_dump(), prob, names, np.array(svals), loan_rec,
    )

    return Response(
        content=bytes(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=credit_assessment_report.pdf"},
    )
