"""
AI-powered Credit Advisor agent with multi-provider support.

Tiers:
  - Free: Gemini 2.0 Flash (app owner's key via env/secrets)
  - Pro:  User's own key — Gemini Pro, GPT-4o, GPT-4o-mini, Claude, etc.

Tools:
  - predict_risk, run_what_if, find_improvements, get_loan_recommendation
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

from src.inference import build_feature_row, load_pipeline, predict_default, shap_values_row

# ---------------------------------------------------------------------------
# Module-level pipeline cache
# ---------------------------------------------------------------------------
_pipeline = None


def _ensure_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline


FEATURE_LABELS = {
    "age": "Age", "MonthlyIncome": "Monthly Income",
    "DebtRatio": "Debt-to-Income Ratio",
    "RevolvingUtilizationOfUnsecuredLines": "Credit Card Utilization",
    "NumberOfTimes90DaysLate": "Times 90+ Days Late",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 Days Late",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 Days Late",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfDependents": "Dependents",
    "total_late_payments": "Total Late Payments",
    "ever_90_days_late": "Ever 90+ Days Late",
    "monthly_debt_estimate": "Monthly Debt Estimate",
    "income_per_dependent": "Income Per Dependent",
    "age_group": "Age Group",
    "high_utilization": "High Credit Utilization Flag",
}

SYSTEM_PROMPT = """\
You are a Credit Risk Advisor AI assistant integrated into a credit scoring \
system powered by an XGBoost machine learning model trained on Kaggle's \
"Give Me Some Credit" dataset.

Your capabilities:
1. Predict default risk for any applicant profile using the real ML model.
2. Explain which factors drive the prediction (via SHAP values).
3. Run what-if scenarios ("what if income was higher?").
4. Suggest concrete improvements to reduce default risk.
5. Provide loan recommendations (amount, rate tier, conditions).

Guidelines:
- Always cite specific numbers and percentages from tool results.
- When explaining factors, reference SHAP contribution values.
- For what-if scenarios, clearly show before vs after comparison.
- Be professional yet accessible; avoid jargon where possible.
- Never make final lending decisions — you provide analysis only.
- If asked about something outside credit risk, politely redirect.
"""


# ---------------------------------------------------------------------------
# Tool implementations (shared by all providers)
# ---------------------------------------------------------------------------

def predict_risk(
    age: float, monthly_income: float, debt_ratio: float,
    revolving_utilization: float, times_30_59_days_late: int,
    times_60_89_days_late: int, times_90_plus_days_late: int,
    open_credit_lines: int, real_estate_loans: int, dependents: int,
) -> dict:
    """Predict the credit default probability for a loan applicant.
    Returns probability, risk level, and top SHAP contributing factors."""
    pipe = _ensure_pipeline()
    X = build_feature_row(
        age=float(age), monthly_income=float(monthly_income),
        debt_ratio=float(debt_ratio), revolving_util=float(revolving_utilization),
        n_30_59_late=float(times_30_59_days_late),
        n_60_89_late=float(times_60_89_days_late),
        n_90_late=float(times_90_plus_days_late),
        n_open_lines=float(open_credit_lines),
        n_real_estate=float(real_estate_loans),
        n_dependents=float(dependents),
    )
    prob, cls = predict_default(pipe, X)
    names, svals, base = shap_values_row(pipe, X)
    order = np.argsort(np.abs(svals))[::-1][:6]
    factors = []
    for idx in order:
        label = FEATURE_LABELS.get(names[idx], names[idx])
        factors.append({
            "feature": label, "raw_name": names[idx],
            "shap_value": round(float(svals[idx]), 4),
            "direction": "increases risk" if svals[idx] > 0 else "decreases risk",
        })
    risk_level = "HIGH" if prob > 0.6 else "MEDIUM" if prob > 0.3 else "LOW"
    return {
        "default_probability": round(prob, 4),
        "default_probability_percent": f"{prob:.1%}",
        "risk_level": risk_level,
        "predicted_class": "DEFAULT" if cls == 1 else "NO DEFAULT",
        "top_factors": factors,
    }


def run_what_if(
    original_age: float, original_monthly_income: float,
    original_debt_ratio: float, original_revolving_utilization: float,
    original_times_30_59_late: int, original_times_60_89_late: int,
    original_times_90_plus_late: int, original_open_credit_lines: int,
    original_real_estate_loans: int, original_dependents: int,
    modified_age: float, modified_monthly_income: float,
    modified_debt_ratio: float, modified_revolving_utilization: float,
    modified_times_30_59_late: int, modified_times_60_89_late: int,
    modified_times_90_plus_late: int, modified_open_credit_lines: int,
    modified_real_estate_loans: int, modified_dependents: int,
) -> dict:
    """Compare original vs modified applicant profile side by side."""
    orig = predict_risk(
        original_age, original_monthly_income, original_debt_ratio,
        original_revolving_utilization, original_times_30_59_late,
        original_times_60_89_late, original_times_90_plus_late,
        original_open_credit_lines, original_real_estate_loans, original_dependents,
    )
    mod = predict_risk(
        modified_age, modified_monthly_income, modified_debt_ratio,
        modified_revolving_utilization, modified_times_30_59_late,
        modified_times_60_89_late, modified_times_90_plus_late,
        modified_open_credit_lines, modified_real_estate_loans, modified_dependents,
    )
    change = mod["default_probability"] - orig["default_probability"]
    return {"original": orig, "modified": mod,
            "probability_change": round(change, 4),
            "probability_change_percent": f"{change:+.1%}", "improved": change < 0}


def find_improvements(
    age: float, monthly_income: float, debt_ratio: float,
    revolving_utilization: float, times_30_59_days_late: int,
    times_60_89_days_late: int, times_90_plus_days_late: int,
    open_credit_lines: int, real_estate_loans: int, dependents: int,
) -> dict:
    """Find top actions to reduce default risk by testing realistic adjustments."""
    pipe = _ensure_pipeline()
    bp = dict(
        age=float(age), monthly_income=float(monthly_income),
        debt_ratio=float(debt_ratio), revolving_util=float(revolving_utilization),
        n_30_59_late=float(times_30_59_days_late),
        n_60_89_late=float(times_60_89_days_late),
        n_90_late=float(times_90_plus_days_late),
        n_open_lines=float(open_credit_lines),
        n_real_estate=float(real_estate_loans),
        n_dependents=float(dependents),
    )
    X_base = build_feature_row(**bp)
    base_prob, _ = predict_default(pipe, X_base)
    adjustments = {
        "revolving_util": ("Reduce credit card utilization", [max(0, revolving_utilization - 0.3), 0.3, 0.1]),
        "debt_ratio": ("Lower debt-to-income ratio", [max(0, debt_ratio - 0.2), 0.3, 0.15]),
        "n_30_59_late": ("Clear 30-59 day late payments", [0]),
        "n_60_89_late": ("Clear 60-89 day late payments", [0]),
        "n_90_late": ("Clear 90+ day late payments", [0]),
        "monthly_income": ("Increase monthly income", [monthly_income * 1.25, monthly_income * 1.5]),
    }
    improvements = []
    for param, (label, values) in adjustments.items():
        for tv in values:
            if abs(tv - bp[param]) < 0.001:
                continue
            mod = bp.copy()
            mod[param] = float(tv)
            X_mod = build_feature_row(**mod)
            new_prob, _ = predict_default(pipe, X_mod)
            reduction = base_prob - new_prob
            if reduction > 0.005:
                improvements.append({
                    "action": label, "parameter": param,
                    "from_value": round(bp[param], 2), "to_value": round(float(tv), 2),
                    "risk_reduction_percent": f"{reduction:.1%}",
                    "new_probability_percent": f"{new_prob:.1%}",
                })
    improvements.sort(key=lambda x: float(x["risk_reduction_percent"].rstrip("%")), reverse=True)
    return {"current_probability_percent": f"{base_prob:.1%}", "top_improvements": improvements[:5]}


def get_loan_recommendation(
    age: float, monthly_income: float, debt_ratio: float,
    revolving_utilization: float, times_30_59_days_late: int,
    times_60_89_days_late: int, times_90_plus_days_late: int,
    open_credit_lines: int, real_estate_loans: int, dependents: int,
) -> dict:
    """Generate loan recommendation with amount, rate tier, and conditions."""
    result = predict_risk(
        age, monthly_income, debt_ratio, revolving_utilization,
        times_30_59_days_late, times_60_89_days_late,
        times_90_plus_days_late, open_credit_lines,
        real_estate_loans, dependents,
    )
    prob = result["default_probability"]
    available = monthly_income * (1 - min(debt_ratio, 0.99))
    max_pmt = available * 0.28
    if prob < 0.2:
        return {"decision": "APPROVE", "risk_tier": "Prime", "rate": "5.5%-7.5%",
                "max_loan": f"${max(0, max_pmt*60):,.0f}", "max_payment": f"${max(0,max_pmt):,.0f}",
                "conditions": ["Standard documentation"], **result}
    elif prob < 0.4:
        return {"decision": "APPROVE WITH CONDITIONS", "risk_tier": "Near-Prime", "rate": "8%-12%",
                "max_loan": f"${max(0, max_pmt*48):,.0f}", "max_payment": f"${max(0,max_pmt):,.0f}",
                "conditions": ["Income proof", "6-month employment check"], **result}
    elif prob < 0.6:
        return {"decision": "CONDITIONAL", "risk_tier": "Subprime", "rate": "13%-18%",
                "max_loan": f"${max(0, max_pmt*36):,.0f}", "max_payment": f"${max(0,max_pmt):,.0f}",
                "conditions": ["Co-signer recommended", "Collateral may be needed"], **result}
    else:
        return {"decision": "HIGH RISK - REVIEW", "risk_tier": "Deep Subprime", "rate": "20%+ or decline",
                "max_loan": f"${max(0, max_pmt*24):,.0f}", "max_payment": f"${max(0,max_pmt):,.0f}",
                "conditions": ["Co-signer required", "Collateral required", "Consider declining"], **result}


# All tools
AGENT_TOOLS = [predict_risk, run_what_if, find_improvements, get_loan_recommendation]

# Tool name -> function mapping
TOOL_MAP = {fn.__name__: fn for fn in AGENT_TOOLS}


# ---------------------------------------------------------------------------
# OpenAI function calling schema
# ---------------------------------------------------------------------------

_PARAM_PROPS = {
    "age": {"type": "number", "description": "Applicant age in years"},
    "monthly_income": {"type": "number", "description": "Monthly income in USD"},
    "debt_ratio": {"type": "number", "description": "Debt-to-income ratio (0-10)"},
    "revolving_utilization": {"type": "number", "description": "Credit card utilization (0-1)"},
    "times_30_59_days_late": {"type": "integer", "description": "Number of times 30-59 days late"},
    "times_60_89_days_late": {"type": "integer", "description": "Number of times 60-89 days late"},
    "times_90_plus_days_late": {"type": "integer", "description": "Number of times 90+ days late"},
    "open_credit_lines": {"type": "integer", "description": "Number of open credit lines"},
    "real_estate_loans": {"type": "integer", "description": "Number of real estate loans"},
    "dependents": {"type": "integer", "description": "Number of dependents"},
}
_REQUIRED = list(_PARAM_PROPS.keys())

_ORIG_PROPS = {f"original_{k}": {**v, "description": f"Original {v['description']}"} for k, v in _PARAM_PROPS.items()}
_MOD_PROPS = {f"modified_{k}": {**v, "description": f"Modified {v['description']}"} for k, v in _PARAM_PROPS.items()}

OPENAI_TOOLS = [
    {"type": "function", "function": {
        "name": "predict_risk", "description": predict_risk.__doc__,
        "parameters": {"type": "object", "properties": _PARAM_PROPS, "required": _REQUIRED}}},
    {"type": "function", "function": {
        "name": "find_improvements", "description": find_improvements.__doc__,
        "parameters": {"type": "object", "properties": _PARAM_PROPS, "required": _REQUIRED}}},
    {"type": "function", "function": {
        "name": "get_loan_recommendation", "description": get_loan_recommendation.__doc__,
        "parameters": {"type": "object", "properties": _PARAM_PROPS, "required": _REQUIRED}}},
    {"type": "function", "function": {
        "name": "run_what_if", "description": run_what_if.__doc__,
        "parameters": {"type": "object",
                        "properties": {**_ORIG_PROPS, **_MOD_PROPS},
                        "required": list(_ORIG_PROPS) + list(_MOD_PROPS)}}},
]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context_prompt(applicant: dict | None) -> str:
    if not applicant:
        return SYSTEM_PROMPT + "\n\nNo applicant scored yet. Ask the user to score one first."
    return SYSTEM_PROMPT + f"""

CURRENT APPLICANT:
- Age: {applicant.get('age', 'N/A')}
- Monthly Income: ${applicant.get('monthly_income', 0):,.0f}
- Debt-to-Income: {applicant.get('debt_ratio', 0):.2f}
- Credit Utilization: {applicant.get('revolving_util', 0):.0%}
- Late 30-59d: {applicant.get('n_30_59_late', 0):.0f}
- Late 60-89d: {applicant.get('n_60_89_late', 0):.0f}
- Late 90+d: {applicant.get('n_90_late', 0):.0f}
- Open Lines: {applicant.get('n_open_lines', 0):.0f}
- RE Loans: {applicant.get('n_real_estate', 0):.0f}
- Dependents: {applicant.get('n_dependents', 0):.0f}

Use these as baseline when calling tools. When user says "this applicant", use these values."""


# ---------------------------------------------------------------------------
# Provider: Gemini
# ---------------------------------------------------------------------------

def create_gemini_session(api_key: str, applicant: dict | None = None, model_name: str = "gemini-2.0-flash"):
    """Create a Gemini chat session with function calling."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        tools=AGENT_TOOLS,
        system_instruction=build_context_prompt(applicant),
    )
    return model.start_chat(enable_automatic_function_calling=True)


# ---------------------------------------------------------------------------
# Provider: OpenAI (GPT-4o, GPT-4o-mini, etc.)
# ---------------------------------------------------------------------------

class OpenAIChatSession:
    """Wrapper around OpenAI chat completions with function calling."""

    def __init__(self, client, model: str, applicant: dict | None = None):
        self.client = client
        self.model = model
        self.messages = [{"role": "system", "content": build_context_prompt(applicant)}]

    def send_message(self, user_message: str):
        """Send message and handle function calling loop."""
        self.messages.append({"role": "user", "content": user_message})

        max_rounds = 8
        for _ in range(max_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                self.messages.append({"role": "assistant", "content": msg.content})
                return _SimpleResponse(msg.content)

            # Execute tool calls
            self.messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                fn = TOOL_MAP.get(fn_name)
                if fn:
                    result = fn(**fn_args)
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}
                self.messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })

        return _SimpleResponse("I'm having trouble processing that. Please try again.")


class _SimpleResponse:
    """Minimal response wrapper to match Gemini's interface."""
    def __init__(self, text: str):
        self.text = text or ""


def create_openai_session(api_key: str, applicant: dict | None = None,
                          model_name: str = "gpt-4o-mini"):
    """Create an OpenAI chat session with function calling."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    return OpenAIChatSession(client, model_name, applicant)


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------

PROVIDER_MODELS = {
    "gemini": {
        "Gemini 2.0 Flash (Free)": "gemini-2.0-flash",
        "Gemini 2.5 Pro": "gemini-2.5-pro-preview-05-06",
        "Gemini 2.5 Flash": "gemini-2.5-flash-preview-04-17",
    },
    "openai": {
        "GPT-4o Mini": "gpt-4o-mini",
        "GPT-4o": "gpt-4o",
        "GPT-4.1": "gpt-4.1",
        "GPT-4.1 Mini": "gpt-4.1-mini",
    },
}


def create_chat_session(provider: str, api_key: str, model_name: str,
                        applicant: dict | None = None):
    """Create a chat session for the given provider.

    Args:
        provider: "gemini", "openai", "openrouter", or "groq"
        api_key: API key for the provider.
        model_name: Model identifier (e.g. "gemini-2.0-flash", "gpt-4o").
        applicant: Current applicant context.

    Returns:
        Chat session with .send_message(text) -> response with .text
    """
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=20.0)
        return OpenAIChatSession(client, model_name, applicant)
    elif provider == "openrouter":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=20.0)
        return OpenAIChatSession(client, model_name, applicant)
    elif provider == "groq":
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=20.0)
        return OpenAIChatSession(client, model_name, applicant)
    else:
        return create_gemini_session(api_key, applicant, model_name)
