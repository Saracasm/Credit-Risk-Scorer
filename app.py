"""
Streamlit UI for credit default risk scoring with SHAP and fairness table.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.inference import (
    build_feature_row,
    load_pipeline,
    predict_default,
    shap_values_row,
)
from src.monitor import check_and_alert, log_prediction

PROJECT_ROOT = Path(__file__).resolve().parent
FAIRNESS_CSV = PROJECT_ROOT / "data" / "fairness_report.csv"
TRAIN_FEATURES = PROJECT_ROOT / "data" / "splits" / "X_train.csv"

# Plain English explanations for each feature
FEATURE_EXPLANATIONS = {
    "age": {
        "label": "Applicant's Age",
        "explanation": lambda val: f"Age {int(val)} years old. Younger borrowers (< 30) and very old ones (> 70) tend to default more.",
    },
    "MonthlyIncome": {
        "label": "Monthly Income",
        "explanation": lambda val: f"Earns ${val:,.0f}/month. Lower income = less ability to absorb financial shocks.",
    },
    "DebtRatio": {
        "label": "Debt-to-Income Ratio",
        "explanation": lambda val: f"Debt ratio of {val:.2f}. This means monthly debt is {val:.0%} of income. Anything over 0.5 (50%) means debt is dangerously high.",
    },
    "RevolvingUtilizationOfUnsecuredLines": {
        "label": "Credit Card Usage",
        "explanation": lambda val: f"Using {val:.0%} of available credit. Over 90% = maxed out = financial stress signal.",
    },
    "NumberOfTimes90DaysLate": {
        "label": "Times 90+ Days Late",
        "explanation": lambda val: f"Has been {int(val)} times severely late (90+ days). Strong signal of payment problems.",
    },
    "NumberOfTime30-59DaysPastDueNotWorse": {
        "label": "Times 30-59 Days Late",
        "explanation": lambda val: f"Has been {int(val)} times moderately late. Shows payment struggle history.",
    },
    "NumberOfTime60-89DaysPastDueNotWorse": {
        "label": "Times 60-89 Days Late",
        "explanation": lambda val: f"Has been {int(val)} times late 60-89 days. Pattern of payment difficulties.",
    },
    "NumberOfOpenCreditLinesAndLoans": {
        "label": "Open Credit Lines",
        "explanation": lambda val: f"Has {int(val)} open credit accounts. More accounts = more debt obligations to manage.",
    },
    "NumberRealEstateLoansOrLines": {
        "label": "Real Estate Loans",
        "explanation": lambda val: f"Has {int(val)} mortgage(s). Real estate debt is serious; missed mortgage = foreclosure.",
    },
    "NumberOfDependents": {
        "label": "Dependents",
        "explanation": lambda val: f"Supports {int(val)} people. More dependents = less money for loan payments.",
    },
    "total_late_payments": {
        "label": "Total Late Payments",
        "explanation": lambda val: f"Total {int(val)} late payment incidents. Pattern of not paying on time = high risk.",
    },
    "ever_90_days_late": {
        "label": "Ever 90+ Days Late",
        "explanation": lambda val: "Has experienced severe delinquency. This is a major red flag for lenders.",
    },
    "monthly_debt_estimate": {
        "label": "Monthly Debt Amount",
        "explanation": lambda val: f"Estimated ${val:,.0f}/month in debt payments. High debt burden limits ability to take new loan.",
    },
    "income_per_dependent": {
        "label": "Income Per Dependent",
        "explanation": lambda val: f"${val:,.0f} per dependent per month. Low = financial pressure; high = comfortable.",
    },
    "age_group": {
        "label": "Age Group",
        "explanation": lambda val: {
            0: "18-30 years old. Younger = fewer financial resources, more job instability.",
            1: "31-45 years old. Mid-career, often peak earning years.",
            2: "46-60 years old. Approaching retirement, may have weaker income growth.",
            3: "60+ years old. Near/at retirement, fixed income may be tight.",
        }.get(int(val), "Unknown age group"),
    },
    "high_utilization": {
        "label": "High Credit Card Usage",
        "explanation": lambda val: "Maxed out credit (>90%). Red flag for financial distress.",
    },
}


@st.cache_resource
def cached_model():
    """
    Load the trained pipeline once per process.

    Returns:
        Fitted sklearn Pipeline.
    """
    return load_pipeline()



def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Credit Risk Scorer",
        page_icon="💳",
        layout="wide",
    )

    # Header with intro
    st.markdown(
        """
        <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">💳 Credit Risk Scorer</h1>
            <p style="color: white; margin: 5px 0 0 0;">See if a loan applicant is likely to default on their debt</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 How It Works")
    st.markdown(
        """
This AI model analyzes an applicant's financial profile and predicts the likelihood they'll default
(fail to pay back a loan). Banks use tools like this to make lending decisions fairly and quickly.

**Green = Low Risk** (likely to pay back) | **Red = High Risk** (may struggle to pay back)
    """
    )

    try:
        pipe = cached_model()
    except Exception as e:
        st.error(
            f"Could not load model from models/xgb_pipeline.pkl. "
            f"Train the model first (python src/train.py). ({e})"
        )
        return

    # Sidebar with clearer labels
    st.sidebar.markdown("### 👤 Applicant Details")
    st.sidebar.markdown("Adjust the sliders below to see how each factor affects default risk:")

    age = st.sidebar.slider("👨‍💼 Age (years)", 18, 100, 45)
    monthly_income = st.sidebar.number_input(
        "💰 Monthly Income ($)", min_value=0.0, max_value=500_000.0, value=5000.0
    )
    debt_ratio = st.sidebar.slider("📊 Debt-to-Income Ratio", 0.0, 10.0, 0.3, help="Higher = more debt relative to income")
    revolving_util = st.sidebar.slider(
        "💳 Credit Card Usage (0-100%)", 0.0, 1.0, 0.2, help="How much of available credit is being used"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Late Payments (how often?)**")
    n_90 = st.sidebar.slider("90+ days late", 0, 20, 0, help="Very serious - shows major payment trouble")
    n_30_59 = st.sidebar.slider("30-59 days late", 0, 20, 0, help="Moderate - shows payment struggles")
    n_60_89 = st.sidebar.slider("60-89 days late", 0, 20, 0, help="Moderate - shows payment struggles")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Other Credit Activity**")
    n_open = st.sidebar.slider(
        "🏦 Open Credit Lines", 0, 50, 5, help="Credit cards, loans, etc."
    )
    n_re = st.sidebar.slider(
        "🏠 Real Estate Loans", 0, 20, 1, help="Mortgages and home loans"
    )
    n_dep = st.sidebar.slider("👨‍👩‍👧‍👦 Dependents", 0, 15, 0, help="People depending on income")

    if st.sidebar.button("🔍 Analyze Applicant", type="primary", use_container_width=True):
        X = build_feature_row(
            age=float(age),
            monthly_income=float(monthly_income),
            debt_ratio=float(debt_ratio),
            revolving_util=float(revolving_util),
            n_30_59_late=float(n_30_59),
            n_60_89_late=float(n_60_89),
            n_90_late=float(n_90),
            n_open_lines=float(n_open),
            n_real_estate=float(n_re),
            n_dependents=float(n_dep),
        )
        prob, pred_cls = predict_default(pipe, X)

        input_dict = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": n_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": n_open,
            "NumberOfTimes90DaysLate": n_90,
            "NumberRealEstateLoansOrLines": n_re,
            "NumberOfTime60-89DaysPastDueNotWorse": n_60_89,
            "NumberOfDependents": n_dep,
        }
        log_prediction(input_dict, pred_cls, prob)

        if TRAIN_FEATURES.exists():
            ref = pd.read_csv(TRAIN_FEATURES)
            check_and_alert(X, ref)

        # Prediction Result - Big and Visual
        st.markdown("---")
        st.markdown("### 🎯 Analysis Result")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### Default Probability")
            if prob > 0.7:
                st.markdown(f'<h1 style="color: #d62728; text-align: center;">{prob:.0%}</h1>', unsafe_allow_html=True)
                st.error("🔴 **HIGH RISK** — This applicant is likely to default")
            elif prob > 0.4:
                st.markdown(f'<h1 style="color: #ff7f0e; text-align: center;">{prob:.0%}</h1>', unsafe_allow_html=True)
                st.warning("🟡 **MEDIUM RISK** — Proceed with caution")
            else:
                st.markdown(f'<h1 style="color: #2ca02c; text-align: center;">{prob:.0%}</h1>', unsafe_allow_html=True)
                st.success("🟢 **LOW RISK** — This applicant looks reliable")

            # Risk meter
            st.markdown("### Risk Meter")
            st.progress(min(max(prob, 0.0), 1.0))

        with col2:
            st.markdown("### Key Factors Affecting This Decision")
            st.markdown(
                "_Red bars increase default risk | Green bars reduce default risk_"
            )
            names, svals, _base = shap_values_row(pipe, X)
            order = np.argsort(np.abs(svals))[-8:]  # Top 8 factors
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in svals[order]]
            ax.barh([names[i] for i in order], svals[order], color=colors)
            ax.set_xlabel("Impact on Default Risk →")
            ax.axvline(0.0, color="gray", linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Detailed factor breakdown
            st.markdown("---")
            st.markdown("### 🔍 Why These Factors Matter")

            # Create a table of top factors with their values and explanations
            factor_details = []
            X_dict = X.to_dict('records')[0]  # Get first row as dict

            for idx in order:
                feat_name = names[idx]
                shap_val = svals[idx]
                risk_direction = "⬆️ INCREASES RISK" if shap_val > 0 else "⬇️ DECREASES RISK"

                if feat_name in FEATURE_EXPLANATIONS:
                    info = FEATURE_EXPLANATIONS[feat_name]
                    explanation_fn = info["explanation"]
                    feat_value = X_dict.get(feat_name, "N/A")
                    explanation = explanation_fn(feat_value)
                else:
                    explanation = f"Value: {X_dict.get(feat_name, 'N/A')}"

                factor_details.append({
                    "Factor": info.get("label", feat_name),
                    "Direction": risk_direction,
                    "Explanation": explanation,
                })

            for i, detail in enumerate(factor_details):
                with st.expander(f"{i+1}. {detail['Factor']} {detail['Direction']}", expanded=i < 3):
                    st.markdown(detail['Explanation'])

        # Plain English Explanation
        st.markdown("---")
        st.markdown("### 📖 This Person's Financial Picture")

        X_dict = X.to_dict('records')[0]

        # Summary narrative
        age_val = int(X_dict.get('age', 0))
        income_val = X_dict.get('MonthlyIncome', 0)
        debt_ratio_val = X_dict.get('DebtRatio', 0)
        late_count = int(X_dict.get('total_late_payments', 0))
        util_val = X_dict.get('RevolvingUtilizationOfUnsecuredLines', 0)

        summary = f"""
**Who:** {age_val}-year-old earning ${income_val:,.0f}/month
"""

        if debt_ratio_val > 2.0:
            summary += f"- 🚨 **Very high debt load** — spends {debt_ratio_val:.0%} of income on debt payments (sustainable is <35%)\n"
        elif debt_ratio_val > 0.5:
            summary += f"- ⚠️ **High debt load** — spends {debt_ratio_val:.0%} of income on existing debt payments\n"
        else:
            summary += f"- ✅ **Manageable debt** — debt is {debt_ratio_val:.0%} of income\n"

        if late_count > 3:
            summary += f"- 🚨 **Serious payment history** — has {late_count} late payment incidents. This is a major red flag\n"
        elif late_count > 0:
            summary += f"- ⚠️ **Some payment issues** — has {late_count} late payments. Shows financial stress in the past.\n"
        else:
            summary += f"- ✅ **Clean payment history** — has never been late. Good sign.\n"

        if util_val > 0.9:
            summary += f"- 🚨 **Maxed out credit** — using {util_val:.0%} of available credit. Financially stressed.\n"
        elif util_val > 0.7:
            summary += f"- ⚠️ **High credit usage** — using {util_val:.0%} of available credit. Limited financial cushion.\n"
        else:
            summary += f"- ✅ **Good credit cushion** — only using {util_val:.0%} of available credit.\n"

        st.markdown(summary)

        if prob > 0.5:
            st.markdown(
                f"""
**🔴 Bottom line:** This person has a **{prob:.0%} chance of defaulting** within 2 years.

This is higher than average. **Why?** Look at the red factors above — their financial situation makes them risky:
- Existing debt obligations are high relative to income
- If they miss a payment once, there's no cushion to catch up
- Past payment problems suggest they struggle when finances get tight

**What a bank would do:**
- ✅ Ask for collateral (car, house, ...) to secure the loan
- ✅ Require a co-signer (someone else guarantees payment)
- ✅ Offer a smaller loan amount
- ❌ **Or decline entirely** — too risky
                """
            )
        else:
            st.markdown(
                f"""
**🟢 Bottom line:** This person has only a **{prob:.0%} chance of defaulting**.

This is **lower than average** ✅. **Why are they reliable?** Look at the green factors above:
- They have enough income to comfortably cover debt payments
- They have a clean payment history (no late payments)
- Their debt load is manageable

**What a bank would do:**
- ✅ **Approve the loan** with confidence
- ✅ Offer **favorable interest rates** (since they're low risk)
- ✅ **Increase loan limits** — they've proven reliable
                """
            )

        # Fairness Information
        st.markdown("---")
        st.markdown("### ⚖️ Fairness Check (Age Groups)")
        st.markdown(
            "_Is this model treating people fairly across different age groups? "
            "Banks must ensure AI decisions aren't biased._"
        )
        if FAIRNESS_CSV.exists():
            fair_df = pd.read_csv(FAIRNESS_CSV, index_col=0)
            st.dataframe(
                fair_df.style.highlight_max(axis=0, color="#ffcccc").highlight_min(axis=0, color="#ccffcc"),
                use_container_width=True,
            )
            st.markdown(
                "_Higher accuracy = better. Watch for big differences between age groups "
                "(that could mean bias)._"
            )
        else:
            st.info("Fairness report not generated yet. Run: `python src/fairness.py`")

        # Model Info (collapsed)
        with st.expander("📚 Technical Details (Model Info)", expanded=False):
            st.markdown(
                """
#### How This Model Works
- **Algorithm:** XGBoost (gradient-boosted decision trees) wrapped in a sklearn Pipeline
- **Training data:** 150,000 real loan applications from Kaggle's "Give Me Some Credit" dataset
- **Accuracy:** ROC-AUC ~0.86 (out of 1.0 = perfect)
- **Class imbalance handling:** Only ~7% of loans defaulted, so the model is weighted to not ignore defaults
- **Explainability:** SHAP values; every prediction shows which factors helped/hurt the score

#### Limitations
- This is a **demo model** trained on public data, not real production data
- Never use alone — always have human judgment in lending decisions
- The dataset is from 2011; lending patterns have changed
- No personally identifiable info is used or stored
                """
            )


if __name__ == "__main__":
    main()
