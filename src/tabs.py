"""
Tab rendering functions for the Streamlit Credit Risk Scorer app.
Each function renders one tab's content.
"""
from __future__ import annotations
import uuid
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from src.inference import build_feature_row, predict_default, shap_values_row
from src.database import (
    save_prediction, get_all_predictions, get_prediction_stats, save_message,
)
from src.knowledge import knowledge_status
from src.monitor import log_prediction, check_and_alert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAIRNESS_CSV = PROJECT_ROOT / "data" / "fairness_report.csv"
TRAIN_FEATURES = PROJECT_ROOT / "data" / "splits" / "X_train.csv"

# Profile templates
TEMPLATES = {
    "Custom": None,
    "\U0001f7e2 Perfect Applicant": dict(age=42, monthly_income=12000, debt_ratio=0.15, revolving_util=0.12, n_30_59=0, n_60_89=0, n_90=0, n_open=8, n_re=2, n_dep=1),
    "\U0001f7e1 Young Professional": dict(age=25, monthly_income=3500, debt_ratio=0.35, revolving_util=0.45, n_30_59=0, n_60_89=0, n_90=0, n_open=3, n_re=0, n_dep=0),
    "\U0001f534 Financial Stress": dict(age=38, monthly_income=2800, debt_ratio=0.85, revolving_util=0.92, n_30_59=3, n_60_89=2, n_90=1, n_open=12, n_re=1, n_dep=3),
    "\U0001f7e1 Retiree": dict(age=67, monthly_income=3200, debt_ratio=0.25, revolving_util=0.15, n_30_59=0, n_60_89=0, n_90=0, n_open=4, n_re=1, n_dep=0),
}

FEATURE_EXPLANATIONS = {
    "age": ("Age", lambda v: f"Age {int(v)}. Younger (<30) and very old (>70) borrowers tend to default more."),
    "MonthlyIncome": ("Monthly Income", lambda v: f"Earns ${v:,.0f}/month. Lower income = less cushion."),
    "DebtRatio": ("Debt-to-Income", lambda v: f"Ratio {v:.2f} — {v:.0%} of income goes to debt."),
    "RevolvingUtilizationOfUnsecuredLines": ("Credit Utilization", lambda v: f"Using {v:.0%} of available credit."),
    "NumberOfTimes90DaysLate": ("90+ Days Late", lambda v: f"{int(v)} severe late payments."),
    "NumberOfTime30-59DaysPastDueNotWorse": ("30-59 Days Late", lambda v: f"{int(v)} moderate late payments."),
    "NumberOfTime60-89DaysPastDueNotWorse": ("60-89 Days Late", lambda v: f"{int(v)} late payments."),
    "NumberOfOpenCreditLinesAndLoans": ("Open Credit Lines", lambda v: f"{int(v)} open accounts."),
    "NumberRealEstateLoansOrLines": ("Real Estate Loans", lambda v: f"{int(v)} mortgage(s)."),
    "NumberOfDependents": ("Dependents", lambda v: f"Supports {int(v)} people."),
    "total_late_payments": ("Total Late Payments", lambda v: f"{int(v)} total late incidents."),
    "ever_90_days_late": ("Ever 90+ Late", lambda v: "Has severe delinquency history."),
    "monthly_debt_estimate": ("Monthly Debt", lambda v: f"~${v:,.0f}/month in debt."),
    "income_per_dependent": ("Income/Dependent", lambda v: f"${v:,.0f} per dependent."),
    "age_group": ("Age Group", lambda v: f"Group {int(v)}."),
    "high_utilization": ("High Utilization", lambda v: "Credit maxed out." if v else "Credit OK."),
}


def _collect_inputs(pipe):
    """Render sidebar inputs and return params when button clicked."""
    st.sidebar.markdown("### \U0001f464 Applicant Details")

    template = st.sidebar.selectbox("Load a profile template:", list(TEMPLATES.keys()))
    t = TEMPLATES[template]

    age = st.sidebar.slider("Age", 18, 100, t["age"] if t else 45,
        help="Applicant's age in years. Younger borrowers (<30) and very elderly (>70) tend to have higher default rates due to less financial stability.")
    income = st.sidebar.number_input("Monthly Income ($)", 0.0, 500000.0, float(t["monthly_income"] if t else 5000),
        help="Gross monthly income before taxes. Typical range: $2,000-$15,000. Higher income = more ability to repay debt.")
    debt = st.sidebar.slider("Debt-to-Income", 0.0, 10.0, t["debt_ratio"] if t else 0.3,
        help="Total monthly debt payments divided by monthly income. E.g., 0.3 = 30% of income goes to debt. Above 0.5 is considered risky. Above 1.0 means debt exceeds income.")
    util = st.sidebar.slider("Credit Utilization", 0.0, 1.0, t["revolving_util"] if t else 0.2,
        help="Percentage of available credit card limit being used. E.g., 0.3 = using 30% of credit limit. Above 0.9 (90%) signals financial distress.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Late Payment History**")
    n90 = st.sidebar.slider("90+ days late", 0, 20, t["n_90"] if t else 0,
        help="Number of times the applicant has been 90+ days late on any payment. This is the most severe delinquency — even 1 is a major red flag.")
    n30 = st.sidebar.slider("30-59 days late", 0, 20, t["n_30_59"] if t else 0,
        help="Number of times 30-59 days past due. A mild sign of payment trouble — missing a payment by a month. Multiple occurrences suggest a pattern.")
    n60 = st.sidebar.slider("60-89 days late", 0, 20, t["n_60_89"] if t else 0,
        help="Number of times 60-89 days past due. More serious than 30-day lates — the borrower struggled to catch up on payments.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Credit Profile**")
    nopen = st.sidebar.slider("Open Credit Lines", 0, 50, t["n_open"] if t else 5,
        help="Total number of open loans and credit cards. Includes auto loans, student loans, credit cards, etc. Typical: 3-10. Too many can indicate over-leveraging.")
    nre = st.sidebar.slider("Real Estate Loans", 0, 20, t["n_re"] if t else 1,
        help="Number of mortgages or home equity loans. 0 = renter, 1-2 = typical homeowner. High values may indicate real estate investment or over-leveraging.")
    ndep = st.sidebar.slider("Dependents", 0, 15, t["n_dep"] if t else 0,
        help="Number of people financially dependent on the applicant (children, elderly parents, etc.). More dependents = less disposable income for loan payments.")

    return dict(age=float(age), monthly_income=float(income), debt_ratio=float(debt),
                revolving_util=float(util), n_30_59_late=float(n30), n_60_89_late=float(n60),
                n_90_late=float(n90), n_open_lines=float(nopen), n_real_estate=float(nre),
                n_dependents=float(ndep))


def _glass_card(content: str) -> str:
    """Wrap content in a glass card div."""
    return f"""<div style="
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-top: 1px solid rgba(255,255,255,0.15);
        border-left: 1px solid rgba(255,255,255,0.15);
        border-radius: 4px;
        padding: 24px;
        margin-bottom: 16px;
    ">{content}</div>"""


def _risk_gauge_html(prob: float) -> str:
    """Circular SVG risk gauge matching Stitch design."""
    pct = int(prob * 100)
    if prob > 0.6:
        color, glow, label, label_bg = "#ffb4ab", "rgba(255,180,171,0.15)", "HIGH RISK", "rgba(255,180,171,0.1)"
    elif prob > 0.3:
        color, glow, label, label_bg = "#ffb964", "rgba(255,185,100,0.15)", "MEDIUM RISK", "rgba(255,185,100,0.1)"
    else:
        color, glow, label, label_bg = "#10b981", "rgba(16,185,129,0.15)", "LOW RISK", "rgba(16,185,129,0.1)"

    dash = prob * 100  # stroke-dasharray percentage
    return f"""<div style="display:flex;flex-direction:column;align-items:center;padding:32px 0;">
        <div style="position:relative;width:220px;height:220px;display:flex;align-items:center;
            justify-content:center;border-radius:50%;border:6px solid rgba(255,255,255,0.05);
            box-shadow: 0 0 50px {glow};">
            <svg viewBox="0 0 36 36" style="position:absolute;width:200px;height:200px;transform:rotate(-90deg);">
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="2.5"/>
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke="{color}" stroke-width="2.5"
                    stroke-dasharray="{dash}, 100" stroke-linecap="round"
                    style="filter: drop-shadow(0 0 6px {color}); transition: stroke-dasharray 1s ease-out;"/>
            </svg>
            <div style="text-align:center;z-index:1;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:48px;font-weight:700;
                    color:white;line-height:1.1;letter-spacing:-0.02em;">{pct}%</div>
                <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
                    color:#8f909e;letter-spacing:0.2em;text-transform:uppercase;margin-top:4px;">
                    DEFAULT RISK</div>
            </div>
        </div>
        <div style="margin-top:20px;display:flex;flex-direction:column;align-items:center;gap:8px;">
            <div style="background:{label_bg};border:1px solid {color}33;
                padding:6px 16px;border-radius:999px;display:inline-flex;align-items:center;gap:6px;">
                <span style="width:8px;height:8px;border-radius:50%;background:{color};
                    box-shadow:0 0 8px {color};"></span>
                <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
                    color:{color};letter-spacing:0.05em;">{label}</span>
            </div>
        </div>
    </div>"""


def _metric_card_html(label: str, value: str, icon: str, sub: str = "", color: str = "#b9c3ff") -> str:
    """Glass metric card matching Stitch layout."""
    sub_html = f'<span style="font-size:12px;color:#10b981;font-family:JetBrains Mono,monospace;margin-left:6px;">{sub}</span>' if sub else ""
    return f"""<div style="
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-top: 1px solid rgba(255,255,255,0.15);
        border-left: 1px solid rgba(255,255,255,0.15);
        border-radius: 4px;
        padding: 20px;
    ">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
                color:#c5c5d5;letter-spacing:0.05em;text-transform:uppercase;">{label}</span>
            <span style="font-size:16px;color:#8f909e;">{icon}</span>
        </div>
        <div style="margin-top:8px;display:flex;align-items:baseline;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:500;
                color:white;">{value}</span>
            {sub_html}
        </div>
    </div>"""


def _shap_bar_html(factor_name: str, shap_val: float, max_abs: float) -> str:
    """Custom SHAP bar matching the Stitch design horizontal bars."""
    is_risk = shap_val > 0
    color = "rgba(255,180,171,0.6)" if is_risk else "rgba(16,185,129,0.6)"
    text_color = "#ffb4ab" if is_risk else "#10b981"
    tag = "Risk" if is_risk else "Positive"
    width_pct = min(abs(shap_val) / max_abs * 45, 45) if max_abs > 0 else 0

    if is_risk:
        bar_html = f'<div style="position:absolute;left:50%;height:8px;width:{width_pct}%;background:{color};border-radius:0 2px 2px 0;"></div>'
    else:
        bar_html = f'<div style="position:absolute;right:50%;height:8px;width:{width_pct}%;background:{color};border-radius:2px 0 0 2px;"></div>'

    return f"""<div style="display:grid;grid-template-columns:140px 1fr 140px;align-items:center;gap:16px;padding:6px 0;">
        <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
            color:#c5c5d5;letter-spacing:0.05em;text-align:right;text-transform:uppercase;">{factor_name}</span>
        <div style="position:relative;height:16px;display:flex;align-items:center;">
            <div style="position:absolute;left:50%;width:1px;height:100%;background:rgba(255,255,255,0.2);"></div>
            {bar_html}
        </div>
        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{text_color};">
            {shap_val:+.2f} ({tag})</span>
    </div>"""


def _empty_gauge_placeholder_html() -> str:
    """Placeholder hero when no score yet (matches dashboard layout)."""
    return """<div style="display:flex;flex-direction:column;align-items:center;padding:28px 0;">
        <div style="position:relative;width:220px;height:220px;display:flex;align-items:center;
            justify-content:center;border-radius:50%;border:6px dashed rgba(255,255,255,0.12);">
            <div style="text-align:center;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:40px;font-weight:700;
                    color:#8f909e;line-height:1.1;">—</div>
                <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
                    color:#8f909e;letter-spacing:0.2em;text-transform:uppercase;margin-top:8px;">
                    RUN ANALYSIS</div>
            </div>
        </div>
        <p style="font-family:'Inter',sans-serif;font-size:13px;color:#8f909e;margin:16px 0 0;max-width:360px;text-align:center;">
            Use the button below to score this profile. You can also use <strong style="color:#b9c3ff;">Analyze Applicant</strong> at the bottom of the sidebar.
        </p>
    </div>"""


def _execute_scorer_analysis(pipe, params: dict) -> None:
    """Train model prediction, persist logs, store result in session."""
    X = build_feature_row(**params)
    prob, cls = predict_default(pipe, X)

    save_prediction(params, prob, cls)

    log_prediction(
        {
            "RevolvingUtilizationOfUnsecuredLines": params["revolving_util"],
            "age": params["age"],
            "DebtRatio": params["debt_ratio"],
            "MonthlyIncome": params["monthly_income"],
            "NumberOfTimes90DaysLate": params["n_90_late"],
            "NumberOfTime30-59DaysPastDueNotWorse": params["n_30_59_late"],
            "NumberOfTime60-89DaysPastDueNotWorse": params["n_60_89_late"],
            "NumberOfOpenCreditLinesAndLoans": params["n_open_lines"],
            "NumberRealEstateLoansOrLines": params["n_real_estate"],
            "NumberOfDependents": params["n_dependents"],
        },
        cls,
        prob,
    )

    if TRAIN_FEATURES.exists():
        try:
            ref = pd.read_csv(TRAIN_FEATURES)
            check_and_alert(X, ref)
        except Exception:
            pass

    st.session_state["last_result"] = {
        "prob": prob,
        "cls": cls,
        "X": X,
        "params": params,
    }


def _render_quick_actions_panel() -> None:
    """Right column: Stitch-style quick actions (shared empty + scored states)."""
    st.markdown(
        _glass_card(
            """
                <div style="margin-bottom:16px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <div>
                            <div style="font-family:'Inter',sans-serif;font-size:11px;font-weight:800;
                                letter-spacing:0.08em;color:#b9c3ff;text-transform:uppercase;">
                                QUICK ACTIONS
                            </div>
                            <div style="font-family:'Inter',sans-serif;font-size:10px;font-weight:800;
                                letter-spacing:0.08em;color:#8f909e;text-transform:uppercase;margin-top:2px;">
                                INSTITUTIONAL GRADE
                            </div>
                        </div>
                        <div style="font-size:20px;">⚡</div>
                    </div>
                    <div style="height:1px;background:rgba(255,255,255,0.06);margin:14px 0;"></div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div style="width:9px;height:9px;border-radius:999px;background:#10b981;box-shadow:0 0 14px rgba(16,185,129,0.35);"></div>
                        <div style="font-family:'Inter',sans-serif;font-size:10px;font-weight:800;color:#8f909e;text-transform:uppercase;letter-spacing:0.06em;">
                            SYSTEM STATUS: NOMINAL
                        </div>
                    </div>
                </div>
                """
        ),
        unsafe_allow_html=True,
    )

    if st.button("\U0001f50d Ask AI Advisor", type="primary", use_container_width=True):
        st.info("Open the **Credit Advisor** tab to chat about this applicant.")
    if st.button("\U0001f504 Run What-If", use_container_width=True):
        st.info("Open the **Scenarios** tab to compare changes side-by-side.")
    if st.button("\U0001f4c4 Generate Report", use_container_width=True):
        st.info("Open the **Report** tab to download the PDF.")
    if st.button("\U0001f6e0 Improve Score", use_container_width=True):
        st.info("Use **Credit Advisor** and ask how to improve the score.")

    st.markdown("---")
    st.caption("Inputs stay in the left sidebar; results update here after analysis.")


def render_scorer_tab(pipe):
    """Render the main risk scorer tab with Stitch-style custom HTML components."""
    params = _collect_inputs(pipe)
    st.session_state["current_applicant"] = params

    sidebar_run = st.sidebar.button(
        "\U0001f50d Analyze Applicant",
        type="primary",
        use_container_width=True,
        help="Scores the applicant using the sidebar values.",
    )

    result = st.session_state.get("last_result")

    if not result:
        # Dashboard-style layout even before first run (matches mockups).
        left_col, right_col = st.columns([2.2, 1.0], gap="large")
        with left_col:
            st.markdown(_glass_card(_empty_gauge_placeholder_html()), unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            p = params
            with c1:
                st.markdown(
                    _metric_card_html(
                        "Monthly Income", f"${p['monthly_income']:,.0f}", "💰"
                    ),
                    unsafe_allow_html=True,
                )
            with c2:
                debt_status = "OPTIMAL" if p["debt_ratio"] < 0.36 else "HIGH"
                st.markdown(
                    _metric_card_html(
                        "Debt-to-Income",
                        f"{p['debt_ratio']:.2f}",
                        "⚖️",
                        debt_status,
                    ),
                    unsafe_allow_html=True,
                )
            with c3:
                util_pct = int(p["revolving_util"] * 100)
                util_status = "HEALTHY" if p["revolving_util"] < 0.3 else "ELEVATED"
                st.markdown(
                    _metric_card_html(
                        "Credit Usage", f"{util_pct}%", "📈", util_status
                    ),
                    unsafe_allow_html=True,
                )
            if st.button(
                "Run risk analysis",
                type="primary",
                use_container_width=True,
                key="main_run_analysis",
            ) or sidebar_run:
                _execute_scorer_analysis(pipe, params)
                st.rerun()
        with right_col:
            _render_quick_actions_panel()
        return

    if sidebar_run:
        _execute_scorer_analysis(pipe, params)
        st.rerun()

    prob, cls, X = result["prob"], result["cls"], result["X"]
    p = result["params"]

    # Match the dashboard screenshot layout: left analytics, right quick actions.
    left_col, right_col = st.columns([2.2, 1.0], gap="large")

    with left_col:
        # --- Hero Risk Gauge (Stitch-style circular gauge) ---
        st.markdown(_glass_card(_risk_gauge_html(prob)), unsafe_allow_html=True)

        # --- Metric Cards Row (Stitch-style glass cards) ---
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                _metric_card_html(
                    "Monthly Income", f"${p['monthly_income']:,.0f}", "💰"
                ),
                unsafe_allow_html=True,
            )
        with c2:
            debt_status = "OPTIMAL" if p["debt_ratio"] < 0.36 else "HIGH"
            st.markdown(
                _metric_card_html(
                    "Debt-to-Income",
                    f"{p['debt_ratio']:.2f}",
                    "⚖️",
                    debt_status,
                ),
                unsafe_allow_html=True,
            )
        with c3:
            util_pct = int(p["revolving_util"] * 100)
            util_status = "HEALTHY" if p["revolving_util"] < 0.3 else "ELEVATED"
            st.markdown(
                _metric_card_html("Credit Usage", f"{util_pct}%", "📈", util_status),
                unsafe_allow_html=True,
            )

        # --- SHAP Factors (Stitch-style custom HTML bars) ---
        names, svals, _base = shap_values_row(pipe, X)
        order = np.argsort(np.abs(svals))[::-1][:6]
        max_abs = max(abs(svals[i]) for i in order) if len(order) > 0 else 1.0

        shap_bars = ""
        for idx in order:
            fn = names[idx]
            lbl = FEATURE_EXPLANATIONS.get(fn, (fn, None))[0]
            shap_bars += _shap_bar_html(lbl, svals[idx], max_abs)

        st.markdown(
            _glass_card(
                f"""
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
                <h3 style="font-family:'Hanken Grotesk',sans-serif;font-weight:600;color:white;
                    margin:0;font-size:18px;">Key Risk Factors</h3>
                <span style="font-family:'Inter',sans-serif;font-size:11px;font-weight:700;
                    color:#8f909e;letter-spacing:0.05em;">MODEL: XGBOOST</span>
            </div>
            <div style="display:flex;flex-direction:column;gap:4px;">
                {shap_bars}
            </div>
        """
            ),
            unsafe_allow_html=True,
        )

        # Factor explanations
        st.markdown("### \U0001f50d Factor Breakdown")
        X_dict = X.to_dict("records")[0]
        order_desc = np.argsort(np.abs(svals))[::-1][:6]
        for i, idx in enumerate(order_desc):
            fn = names[idx]
            sv = svals[idx]
            direction = "\u2b06\ufe0f RISK" if sv > 0 else "\u2b07\ufe0f SAFE"
            lbl, expl_fn = FEATURE_EXPLANATIONS.get(
                fn, (fn, lambda v: f"Value: {v}")
            )
            val = X_dict.get(fn, 0)
            with st.expander(f"{i+1}. {lbl} — {direction}", expanded=i < 3):
                st.markdown(expl_fn(val))

        # Fairness
        st.markdown("### \u2696\ufe0f Fairness Check")
        if FAIRNESS_CSV.exists():
            st.dataframe(
                pd.read_csv(FAIRNESS_CSV, index_col=0), use_container_width=True
            )
        else:
            st.info("Run `python src/fairness.py` to generate fairness report.")

    with right_col:
        _render_quick_actions_panel()


def _get_free_tier_key() -> str | None:
    """Get the free tier Gemini API key from secrets or environment."""
    # 1. Streamlit secrets (for deployed apps)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 2. Environment variable (for local dev)
    import os
    return os.environ.get("GEMINI_API_KEY")


def render_advisor_tab(pipe):
    """Render the AI Credit Advisor chatbot tab with Free / Pro tiers."""
    from src.agent import create_chat_session, PROVIDER_MODELS

    st.markdown("### \U0001f916 Credit Advisor Agent")
    st.markdown("Chat with an AI advisor that can **explain risk scores**, "
                "**run what-if scenarios**, **suggest improvements**, and "
                "**recommend loan terms** — all powered by your real ML model.")

    # --- Tier selection ---
    st.markdown("---")
    tier = st.radio(
        "Choose your plan:",
        ["\u2728 Free Tier (Gemini Flash)", "\U0001f680 Pro Tier (Bring Your Own Key)"],
        horizontal=True,
        help="Free tier uses Gemini 2.0 Flash. Pro tier lets you use GPT-4o, Gemini Pro, etc.",
    )

    api_key = None
    provider = "gemini"
    model_name = "gemini-2.0-flash"

    if "Free" in tier:
        # Free tier — use pre-configured key
        api_key = _get_free_tier_key()
        if not api_key:
            st.warning(
                "**Free tier not configured yet.** The app owner needs to set a Gemini API key.\n\n"
                "**Quick setup:**\n"
                "1. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey)\n"
                "2. Set it as an environment variable: `set GEMINI_API_KEY=your_key_here`\n"
                "3. Or create `.streamlit/secrets.toml` with: `GEMINI_API_KEY = \"your_key\"`\n\n"
                "Or switch to **Pro Tier** and enter your own key below."
            )
            return
        st.success("Using **Gemini 2.0 Flash** (free)")
        provider = "gemini"
        model_name = "gemini-2.0-flash"
    else:
        # Pro tier — user provides their own key
        col_prov, col_model = st.columns(2)
        with col_prov:
            provider = st.selectbox("Provider", ["gemini", "openai"], format_func=lambda x: x.upper())
        with col_model:
            models = PROVIDER_MODELS.get(provider, {})
            model_label = st.selectbox("Model", list(models.keys()))
            model_name = models[model_label]

        api_key = st.text_input(
            f"\U0001f511 {'Gemini' if provider == 'gemini' else 'OpenAI'} API Key",
            type="password",
            help="Your key is never stored — it's only used for this session.",
        )
        if not api_key:
            st.info(
                f"Enter your **{provider.upper()}** API key above.\n\n"
                + ("Get one free at [Google AI Studio](https://aistudio.google.com/apikey)."
                   if provider == "gemini"
                   else "Get one at [OpenAI Platform](https://platform.openai.com/api-keys).")
            )
            return
        st.success(f"Using **{model_label}** via {provider.upper()}")

    # --- Initialize / reset chat ---
    needs_init = "advisor_chat" not in st.session_state
    config_changed = (
        st.session_state.get("_adv_provider") != provider
        or st.session_state.get("_adv_model") != model_name
    )
    if needs_init or config_changed or st.button("\U0001f504 New Conversation"):
        try:
            applicant = st.session_state.get("current_applicant")
            st.session_state.advisor_chat = create_chat_session(
                provider, api_key, model_name, applicant
            )
            st.session_state.advisor_messages = []
            st.session_state.advisor_session_id = str(uuid.uuid4())
            st.session_state._adv_provider = provider
            st.session_state._adv_model = model_name
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")
            return

    # --- Chat display ---
    for msg in st.session_state.get("advisor_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Handle quick action button prompts ---
    quick_prompt = st.session_state.pop("_advisor_prompt", None)

    # --- Chat input (manual or from quick button) ---
    typed_prompt = st.chat_input("Ask about this applicant...")
    prompt = quick_prompt or typed_prompt

    if prompt:
        st.session_state.advisor_messages.append({"role": "user", "content": prompt})
        save_message(st.session_state.advisor_session_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = st.session_state.advisor_chat.send_message(prompt)
                    reply = response.text
                except Exception as e:
                    err = str(e)
                    if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                        reply = (
                            "\u26a0\ufe0f **Rate limit reached.** The free tier has usage limits.\n\n"
                            "**What you can do:**\n"
                            "- Wait ~15 seconds and try again\n"
                            "- Switch to **Pro Tier** with your own API key for higher limits\n"
                            "- Try a shorter question"
                        )
                    else:
                        reply = f"\u274c **Error:** {err[:200]}"
            st.markdown(reply)
            st.session_state.advisor_messages.append({"role": "assistant", "content": reply})
            save_message(st.session_state.advisor_session_id, "assistant", reply)

    # --- Quick action buttons ---
    st.markdown("---")
    st.markdown("**Quick questions:**")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("\U0001f4a1 Why this score?"):
            st.session_state["_advisor_prompt"] = "Why did this applicant get this risk score? Explain the top factors."
            st.rerun()
    with c2:
        if st.button("\U0001f4c8 How to improve?"):
            st.session_state["_advisor_prompt"] = "What are the top 3 things this applicant can do to reduce their default risk?"
            st.rerun()
    with c3:
        if st.button("\U0001f4b0 Loan recommendation"):
            st.session_state["_advisor_prompt"] = "Based on this profile, what loan would you recommend? Include amount, rate, and conditions."
            st.rerun()


def render_scenarios_tab(pipe):
    """Render the What-If Scenario Comparison tab."""
    st.markdown("### \U0001f504 What-If Scenario Comparison")
    st.markdown("Compare up to 3 scenarios side-by-side to see how changes affect risk.")

    base = st.session_state.get("current_applicant", {})
    if not base:
        st.info("Score an applicant in the **Risk Scorer** tab first.")
        return

    st.markdown(f"**Baseline:** Age {base['age']:.0f}, Income ${base['monthly_income']:,.0f}, "
                f"Debt Ratio {base['debt_ratio']:.2f}")

    cols = st.columns(3)
    scenarios = []
    labels = ["Scenario A (Higher Income)", "Scenario B (Lower Debt)", "Scenario C (No Late Payments)"]
    defaults = [
        {"monthly_income": base["monthly_income"] * 1.5},
        {"debt_ratio": max(0.1, base["debt_ratio"] * 0.5)},
        {"n_30_59_late": 0, "n_60_89_late": 0, "n_90_late": 0},
    ]

    for i, (col, label, dflt) in enumerate(zip(cols, labels, defaults)):
        with col:
            st.markdown(f"**{label}**")
            mod = base.copy()
            mod.update(dflt)
            for k, v in dflt.items():
                mod[k] = st.number_input(f"{k}", value=float(v), key=f"sc_{i}_{k}")
            scenarios.append(mod)

    if st.button("Compare Scenarios", type="primary"):
        X_base = build_feature_row(**base)
        prob_base, _ = predict_default(pipe, X_base)

        cols2 = st.columns(3)
        for i, (col, mod, label) in enumerate(zip(cols2, scenarios, labels)):
            with col:
                X_mod = build_feature_row(**mod)
                prob_mod, _ = predict_default(pipe, X_mod)
                change = prob_mod - prob_base
                color = "#34d399" if change < 0 else "#f87171"
                st.metric(label.split("(")[0], f"{prob_mod:.0%}", f"{change:+.1%}",
                          delta_color="inverse")

                names, svals, _ = shap_values_row(pipe, X_mod)
                order = np.argsort(np.abs(svals))[-5:]
                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.barh([names[j] for j in order], svals[order],
                        color=["#ffb4ab" if v > 0 else "#10b981" for v in svals[order]],
                        edgecolor="none", height=0.55, alpha=0.8)
                ax.axvline(0, color="#444653", lw=0.5)
                ax.tick_params(labelsize=7)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown(f"**Baseline risk: {prob_base:.0%}**")


def render_batch_tab(pipe):
    """Render the Batch CSV Scoring tab."""
    st.markdown("### \U0001f4cb Batch CSV Scoring")
    st.markdown("Upload a CSV of applicants to score them all at once.")

    # Template download
    template_df = pd.DataFrame([{
        "age": 45, "MonthlyIncome": 5000, "DebtRatio": 0.3,
        "RevolvingUtilizationOfUnsecuredLines": 0.2,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfDependents": 1,
    }])
    st.download_button("\u2b07\ufe0f Download CSV Template", template_df.to_csv(index=False),
                       "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload applicant CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df)} applicants**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Score All", type="primary"):
            from src.feature_engineering import add_engineered_features
            with st.spinner(f"Scoring {len(df)} applicants..."):
                df_eng = add_engineered_features(df)
                probs = pipe.predict_proba(df_eng)[:, 1]
                df["default_probability"] = probs
                df["risk_level"] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                           labels=["LOW", "MEDIUM", "HIGH"])
                df["predicted_class"] = (probs >= 0.5).astype(int)

            st.success(f"Scored {len(df)} applicants!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Risk", f"{probs.mean():.0%}")
            c2.metric("High Risk", f"{(probs > 0.6).sum()}")
            c3.metric("Low Risk", f"{(probs < 0.3).sum()}")

            st.dataframe(df.style.background_gradient(subset=["default_probability"], cmap="RdYlGn_r"),
                         use_container_width=True)
            st.download_button("\u2b07\ufe0f Download Results", df.to_csv(index=False),
                               "scored_applicants.csv", "text/csv")


def render_dashboard_tab(pipe):
    """Render the Portfolio Risk Dashboard tab."""
    st.markdown("### \U0001f4ca Portfolio Risk Dashboard")

    preds = get_all_predictions(limit=1000)
    stats = get_prediction_stats()

    if preds.empty or stats.get("total", 0) == 0:
        st.info("No predictions yet. Score some applicants to see analytics here.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scored", stats["total"])
    c2.metric("Avg Risk", f"{stats['avg_prob']:.0%}")
    c3.metric("High Risk", stats["high_count"])
    c4.metric("Low Risk", stats["low_count"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Risk Distribution")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(preds["probability"].dropna(), bins=20, color="#7189f6", edgecolor="#121319", alpha=0.8)
        ax.set_xlabel("Default Probability")
        ax.set_ylabel("Count")
        ax.axvline(0.5, color="#ffb4ab", ls="--", alpha=0.7, label="Threshold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Risk Level Breakdown")
        risk_counts = preds["risk_level"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = {"LOW": "#10b981", "MEDIUM": "#ffb964", "HIGH": "#ffb4ab"}
        wedges, texts, autotexts = ax.pie(
            risk_counts.values, labels=risk_counts.index,
            colors=[colors.get(l, "#666") for l in risk_counts.index],
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"edgecolor": "#121319", "linewidth": 2})
        for t in texts: t.set_color("#c5c5d5")
        for t in autotexts: t.set_color("white"); t.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Recent predictions table
    st.markdown("#### Recent Predictions")
    display_cols = [c for c in ["timestamp", "age", "monthly_income", "probability", "risk_level"]
                    if c in preds.columns]
    st.dataframe(preds[display_cols].head(20), use_container_width=True)

    # Policy knowledge base status
    st.markdown("---")
    st.caption(f"Policy knowledge base (RAG): {knowledge_status()}")


def render_report_tab(pipe):
    """Render the PDF Report Generator tab."""
    st.markdown("### \U0001f4c4 Credit Assessment Report")

    result = st.session_state.get("last_result")
    if not result:
        st.info("Score an applicant in the **Risk Scorer** tab first to generate a report.")
        return

    st.markdown(f"**Applicant:** Age {result['params']['age']:.0f}, "
                f"Income ${result['params']['monthly_income']:,.0f}, "
                f"Risk: {result['prob']:.0%}")

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            from src.report_generator import generate_report
            from src.agent import get_loan_recommendation

            names, svals, _ = shap_values_row(pipe, result["X"])
            p = result["params"]
            loan_rec = get_loan_recommendation(
                p["age"], p["monthly_income"], p["debt_ratio"], p["revolving_util"],
                int(p["n_30_59_late"]), int(p["n_60_89_late"]), int(p["n_90_late"]),
                int(p["n_open_lines"]), int(p["n_real_estate"]), int(p["n_dependents"]),
            )
            pdf_bytes = generate_report(p, result["prob"], names, np.array(svals), loan_rec)

        st.success("Report generated!")
        st.download_button("\u2b07\ufe0f Download PDF Report", pdf_bytes,
                           "credit_assessment_report.pdf", "application/pdf")
