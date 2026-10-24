import streamlit as st, pandas as pd, joblib
from src.features import add_features

st.set_page_config(page_title="EMIPredict AI", page_icon="💰", layout="wide")

@st.cache_resource
def load_models():
    return (joblib.load("models/clf_best.joblib"),
            joblib.load("models/reg_best.joblib"),
            joblib.load("models/label_encoder.joblib"))

clf, reg, le = load_models()

st.title("💰 EMIPredict AI")
st.caption("EMI eligibility aur maximum safe monthly EMI ka instant assessment")

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
        st.error("Years of employment age ke hisaab se zyada hain."); st.stop()
    if dependents >= family:
        st.error("Dependents family size se kam hone chahiye."); st.stop()

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

    msg = {"Eligible": "Low risk: EMI comfortably affordable hai.",
           "High_Risk": "Marginal case: loan possible hai, par higher interest rate lag sakta hai.",
           "Not_Eligible": "High risk: is request par loan recommend nahi hai."}[label]
    {"Eligible": st.success, "High_Risk": st.warning, "Not_Eligible": st.error}[label](msg)

    st.bar_chart(pd.Series(proba, index=le.classes_, name="probability"))
