"""
CodeAlpha_CreditScoringModel — Real-Time Prediction GUI
========================================================
Streamlit web app that loads the trained pipeline (credit_model_pipeline.pkl)
produced by train_model.py and lets a user enter an applicant's financial
and personal data to get a live creditworthiness prediction.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load trained pipeline + metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("credit_model_pipeline.pkl"):
        return None, None
    pipeline = joblib.load("credit_model_pipeline.pkl")
    with open("credit_model_metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata

pipeline, metadata = load_artifacts()

if pipeline is None:
    st.error("Model file not found. Please run `python train_model.py` first "
              "to download the dataset and train the model.")
    st.stop()

labels = metadata["category_labels"]
num_ranges = metadata["numeric_ranges"]

def label_select(field, default_code):
    """Show a friendly selectbox for a categorical field, return the underlying code."""
    options = labels[field]
    codes = list(options.keys())
    display = [options[c] for c in codes]
    default_index = codes.index(default_code) if default_code in codes else 0
    choice = st.selectbox(field_titles[field], display, index=default_index)
    # map back from friendly label to code
    return codes[display.index(choice)]

field_titles = {
    "CheckingStatus": "Checking Account Status",
    "CreditHistory": "Credit History",
    "Purpose": "Loan Purpose",
    "Savings": "Savings Account Balance",
    "Employment": "Employment Duration",
    "PersonalStatus": "Personal Status & Sex",
    "OtherDebtors": "Other Debtors / Guarantors",
    "Property": "Property Owned",
    "OtherInstallmentPlans": "Other Installment Plans",
    "Housing": "Housing Situation",
    "Job": "Job / Skill Level",
    "Telephone": "Telephone",
    "ForeignWorker": "Foreign Worker",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💳 Credit Scoring Model")
st.caption("CodeAlpha ML Internship — Task 1: Credit Scoring Model")

with st.expander("ℹ️ About this model"):
    st.write(
        f"**Best model selected:** {metadata['best_model_name']}\n\n"
        f"**Trained on:** Statlog German Credit Dataset (UCI ML Repository)\n\n"
        f"**Test-set performance:**"
    )
    m = metadata["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
    c2.metric("Precision", f"{m['Precision']*100:.1f}%")
    c3.metric("Recall", f"{m['Recall']*100:.1f}%")
    c4.metric("F1-Score", f"{m['F1-Score']*100:.1f}%")
    c5.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")

st.divider()
st.subheader("Enter Applicant Data")
st.write("Fill in the applicant's financial and personal details, then click **Predict** for a real-time credit risk assessment.")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    st.markdown("**Financial details**")
    col1, col2 = st.columns(2)
    with col1:
        checking = label_select("CheckingStatus", "A12")
        credit_history = label_select("CreditHistory", "A32")
        purpose = label_select("Purpose", "A43")
        savings = label_select("Savings", "A61")
        duration = st.number_input("Loan Duration (months)", min_value=1, max_value=120, value=24)
        credit_amount = st.number_input("Credit Amount (DM)", min_value=0, max_value=100000, value=3000)
        installment_rate = st.slider("Installment Rate (% of disposable income)", 1, 4, 3)

    with col2:
        employment = label_select("Employment", "A73")
        other_debtors = label_select("OtherDebtors", "A101")
        property_owned = label_select("Property", "A121")
        other_plans = label_select("OtherInstallmentPlans", "A143")
        housing = label_select("Housing", "A152")
        existing_credits = st.slider("Number of Existing Credits at this Bank", 1, 4, 1)
        residence_since = st.slider("Years at Current Residence", 1, 4, 2)

    st.markdown("**Personal details**")
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        personal_status = label_select("PersonalStatus", "A93")
        job = label_select("Job", "A173")

    with col4:
        num_liable = st.slider("Number of People Financially Dependent", 1, 2, 1)
        telephone = label_select("Telephone", "A192")
        foreign_worker = label_select("ForeignWorker", "A201")

    submitted = st.form_submit_button("🔍 Predict Creditworthiness", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    input_dict = {
        "CheckingStatus": checking,
        "Duration": duration,
        "CreditHistory": credit_history,
        "Purpose": purpose,
        "CreditAmount": credit_amount,
        "Savings": savings,
        "Employment": employment,
        "InstallmentRate": installment_rate,
        "PersonalStatus": personal_status,
        "OtherDebtors": other_debtors,
        "ResidenceSince": residence_since,
        "Property": property_owned,
        "Age": age,
        "OtherInstallmentPlans": other_plans,
        "Housing": housing,
        "ExistingCredits": existing_credits,
        "Job": job,
        "NumLiable": num_liable,
        "Telephone": telephone,
        "ForeignWorker": foreign_worker,
    }

    ordered_cols = metadata["numeric_features"] + metadata["categorical_features"]
    input_df = pd.DataFrame([input_dict])[ordered_cols]

    prediction = pipeline.predict(input_df)[0]
    probability_good = pipeline.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ **Good Credit Risk** — estimated probability: {probability_good*100:.1f}%")
    else:
        st.error(f"⚠️ **Bad Credit Risk** — estimated probability of good credit: {probability_good*100:.1f}%")

    st.progress(min(float(probability_good), 1.0))
    st.caption(
        "This is a machine learning estimate based on the Statlog German Credit "
        "dataset and is **not a financial or lending decision**. Please consult "
        "a certified financial advisor or loan officer for actual credit decisions."
    )

    with st.expander("View input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Built by Raza Ahmad Khan · CodeAlpha Machine Learning Internship · Task 1")
