import streamlit as st, pandas as pd, joblib
from src.features import add_features

st.set_page_config(page_title="EMIPredict AI", page_icon="💰", layout="wide")

@st.cache_resource
def load_models():
    return (joblib.load("models/clf_best.joblib"),
            joblib.load("models/reg_best.joblib"),
            joblib.load("models/label_encoder.joblib"))

clf, reg, le = load_models()

RANGES = {  # (amount_min, amount_max, tenure_min, tenure_max) from the training data
    "E-commerce Shopping EMI": (10_000, 200_000, 3, 24),
    "Home Appliances EMI":     (20_000, 300_000, 6, 36),
    "Vehicle EMI":             (80_000, 1_500_000, 12, 84),
    "Personal Loan EMI":       (50_000, 1_000_000, 12, 60),
    "Education EMI":           (50_000, 500_000, 6, 48),
}

st.title("💰 EMIPredict AI")
st.caption("Instant assessment of EMI eligibility and the maximum safe monthly EMI")

with st.form("emi_form"):
    st.subheader("Personal")
    a, b, c, d = st.columns(4)
    age = a.number_input("Age", 25, 60, 32)
    gender = b.selectbox("Gender", ["Male", "Female"])
    marital = c.selectbox("Marital status", ["Single", "Married"])
    education = d.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])

    st.subheader("Employment & Housing")
    a, b, c, d = st.columns(4)
    salary = a.number_input("Monthly salary (INR)", 15000, 200000, 60000, step=1000)
    emp = b.selectbox("Employment type", ["Private", "Government", "Self-employed"])
    years = c.number_input("Years of employment", 0, 40, 5)
    company = d.selectbox("Company type", ["Small", "Mid-size", "Large", "MNC", "Startup"])
    a, b, c, d = st.columns(4)
    house = a.selectbox("House type", ["Rented", "Own", "Family"])
    rent = b.number_input("Monthly rent", 0, 100000, 12000, step=500)
    family = c.number_input("Family size", 1, 10, 4)
    dependents = d.number_input("Dependents", 0, 9, 1)

    st.subheader("Monthly expenses")
    a, b, c, d, e = st.columns(5)
    school = a.number_input("School fees", 0, 50000, 0, step=500)
    college = b.number_input("College fees", 0, 50000, 0, step=500)
    travel = c.number_input("Travel", 0, 50000, 3000, step=500)
    groceries = d.number_input("Groceries & utilities", 0, 100000, 10000, step=500)
    other = e.number_input("Other expenses", 0, 100000, 4000, step=500)

    st.subheader("Credit & Savings")
    a, b, c, d, e = st.columns(5)
    existing = a.selectbox("Existing loans", ["No", "Yes"])
    cur_emi = b.number_input("Current EMI amount", 0, 100000, 0, step=500)
    credit = c.number_input("Credit score", 300, 850, 700)
    bank = d.number_input("Bank balance", 0, 5000000, 200000, step=5000)
    emerg = e.number_input("Emergency fund", 0, 2000000, 100000, step=5000)

    st.subheader("Loan request")
    a, b, c = st.columns(3)
    scenario = a.selectbox("EMI scenario", ["E-commerce Shopping EMI", "Home Appliances EMI",
                                            "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
    amount = b.number_input("Requested amount (INR)", 10000, 1500000, 200000, step=5000)
    tenure = c.number_input("Tenure (months)", 3, 84, 24)

    go = st.form_submit_button("Assess", type="primary")

if go:
    if years > age - 18:
        st.error("Years of employment cannot exceed your working-age years."); st.stop()
    if dependents >= family:
        st.error("Dependents must be fewer than the family size."); st.stop()

    a_lo, a_hi, t_lo, t_hi = RANGES[scenario]
    if not (a_lo <= amount <= a_hi) or not (t_lo <= tenure <= t_hi):
        st.warning(f"⚠️ For {scenario}, the training data covers amounts from ₹{a_lo:,} to ₹{a_hi:,} "
                   f"and tenures from {t_lo} to {t_hi} months. Predictions outside this range are less reliable.")

    row = pd.DataFrame([{
        "age": age, "gender": gender, "marital_status": marital, "education": education,
        "monthly_salary": salary, "employment_type": emp, "years_of_employment": years,
        "company_type": company, "house_type": house, "monthly_rent": rent,
        "family_size": family, "dependents": dependents, "school_fees": school,
        "college_fees": college, "travel_expenses": travel, "groceries_utilities": groceries,
        "other_monthly_expenses": other, "existing_loans": existing,
        "current_emi_amount": cur_emi, "credit_score": credit, "bank_balance": bank,
        "emergency_fund": emerg, "emi_scenario": scenario,
        "requested_amount": amount, "requested_tenure": tenure}])
    X = add_features(row)

    label = le.inverse_transform(clf.predict(X))[0]
    proba = clf.predict_proba(X)[0]
    max_emi = float(reg.predict(X)[0])
    req_emi = amount / tenure

    c1, c2, c3 = st.columns(3)
    c1.metric("Eligibility", label.replace("_", " "))
    c2.metric("Max safe monthly EMI", f"₹{max_emi:,.0f}")
    c3.metric("Requested EMI", f"₹{req_emi:,.0f}", delta=f"{(max_emi - req_emi):,.0f} margin")

    msg = {"Eligible": "Low risk: the EMI is comfortably affordable.",
           "High_Risk": "Marginal case: the loan is possible, but a higher interest rate may apply.",
           "Not_Eligible": "High risk: this loan request is not recommended."}[label]
    {"Eligible": st.success, "High_Risk": st.warning, "Not_Eligible": st.error}[label](msg)

    st.bar_chart(pd.Series(proba, index=le.classes_, name="probability"))
