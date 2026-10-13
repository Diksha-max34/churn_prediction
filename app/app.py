"""Streamlit app: score a customer's churn risk and explain it with SHAP.
Run:  streamlit run app/app.py   (after `python -m src.train`)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="Churn Risk Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_artifacts():
    art = joblib.load(ROOT / "models" / "churn_model.joblib")
    art["explainer"] = shap.TreeExplainer(art["model"])
    return art


art = load_artifacts()
metrics = json.load(open(ROOT / "reports" / "metrics.json"))
cats = art["categories"]

st.title("📉 Customer Churn Risk Predictor")
st.caption("XGBoost + SHAP explainability on the IBM Telco customer churn dataset.")

tab_pred, tab_model = st.tabs(["Predict a customer", "About the model"])

# ---------------- Prediction tab ----------------
with tab_pred:
    with st.sidebar:
        st.header("Customer profile")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        inputs = {"tenure": tenure, "MonthlyCharges": monthly, "SeniorCitizen": senior,
                  "TotalCharges": tenure * monthly}  # approximation for a quick demo
        for c in art["categorical"]:
            inputs[c] = st.selectbox(c, sorted(cats[c]))

    row = pd.DataFrame([inputs])[art["numeric"] + art["categorical"]]
    X = art["preprocessor"].transform(row)
    proba = float(art["model"].predict_proba(X)[0, 1])
    thr = art["threshold"]
    high = proba >= thr

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{proba:.1%}")
    c2.metric("Decision threshold", f"{thr:.2f}")
    c3.metric("Risk level", "HIGH — target for retention" if high else "Lower risk")
    (st.error if high else st.success)(
        "Recommended action: include in retention campaign." if high
        else "No retention offer recommended at current cost assumptions.")

    st.subheader("Why this prediction? (SHAP)")
    sv = art["explainer"].shap_values(X)[0]
    exp = shap.Explanation(values=sv, base_values=art["explainer"].expected_value,
                           data=X[0], feature_names=art["feature_names"])
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red bars push churn risk up, blue bars push it down (log-odds scale).")

# ---------------- Model tab ----------------
with tab_model:
    st.subheader("Test-set performance (held-out 20%)")
    xgb, lr = metrics["xgboost"], metrics["logistic_regression_baseline"]
    st.dataframe(pd.DataFrame({
        "Logistic Regression (baseline)": lr["cost_optimised"],
        "XGBoost (final)": xgb["cost_optimised"]}).T, width="stretch")
    st.write(f"Net benefit on test set — XGBoost: **${xgb['net_benefit_$']:,.0f}**, "
             f"baseline: **${lr['net_benefit_$']:,.0f}**, contact everyone: "
             f"**${metrics['no_model_contact_everyone_net_benefit_$']:,.0f}**.")
    st.caption("Assumptions: " + ", ".join(f"{k}={v}" for k, v in metrics["assumptions"].items()))
    a, b = st.columns(2)
    a.image(str(ROOT / "reports" / "shap_summary.png"), caption="Global SHAP summary")
    b.image(str(ROOT / "reports" / "threshold_curve.png"), caption="Cost-based threshold")
