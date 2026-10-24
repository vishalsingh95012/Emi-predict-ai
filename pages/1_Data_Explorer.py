import streamlit as st, pandas as pd
st.title("📊 Data Explorer")

df = pd.read_csv("data/sample.csv")
st.caption(f"{len(df):,} records ka sample (original dataset: 400K)")

opts = list(df["emi_scenario"].unique())
sc = st.multiselect("EMI scenario", opts, default=opts)
d = df[df["emi_scenario"].isin(sc)]

c1, c2 = st.columns(2)
c1.subheader("Eligibility by scenario")
c1.bar_chart(pd.crosstab(d["emi_scenario"], d["emi_eligibility"]))
c2.subheader("Eligibility by employment type")
c2.bar_chart(pd.crosstab(d["employment_type"], d["emi_eligibility"], normalize="index"))

st.subheader("Credit score vs max EMI")
st.scatter_chart(d.sample(min(3000, len(d)), random_state=1),
                 x="credit_score", y="max_monthly_emi", color="emi_eligibility")

st.subheader("Statistics")
st.dataframe(d.describe().T)
