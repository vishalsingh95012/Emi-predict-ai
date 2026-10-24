import pandas as pd

def add_features(d):
    d = d.copy()
    s = d["monthly_salary"]
    d["total_expenses"] = d[["monthly_rent", "school_fees", "college_fees", "travel_expenses",
                             "groceries_utilities", "other_monthly_expenses"]].sum(axis=1)
    d["requested_emi"] = d["requested_amount"] / d["requested_tenure"]
    d["expense_to_income"] = d["total_expenses"] / s
    d["existing_emi_to_income"] = d["current_emi_amount"] / s
    d["requested_emi_to_income"] = d["requested_emi"] / s
    d["dti_after_loan"] = (d["current_emi_amount"] + d["requested_emi"]) / s
    d["disposable_income"] = s - d["total_expenses"] - d["current_emi_amount"]
    d["affordability_ratio"] = d["requested_emi"] / d["disposable_income"].clip(lower=1)
    d["savings_to_salary"] = (d["bank_balance"] + d["emergency_fund"]) / s
    d["employment_stability"] = d["years_of_employment"] / (d["age"] - 20)
    d["credit_risk_band"] = pd.cut(d["credit_score"], [0, 580, 670, 740, 900],
                                   labels=["Poor", "Fair", "Good", "Excellent"]).astype(str)
    d["credit_x_disposable"] = d["credit_score"] * d["disposable_income"] / 1e5
    d["dependents_x_expense"] = d["dependents"] * d["expense_to_income"]
    return d
