import streamlit as st, pandas as pd
st.title("📈 Model Performance (MLflow)")

clf = pd.read_csv("clf_comparison.csv"); reg = pd.read_csv("reg_comparison.csv")

st.subheader("Classification (validation)")
st.dataframe(clf, hide_index=True)
st.bar_chart(clf.set_index("model")[["val_accuracy", "val_f1", "val_roc_auc"]])

st.subheader("Regression (validation)")
st.dataframe(reg, hide_index=True)
st.bar_chart(reg.set_index("model")[["val_rmse"]])

st.success("Best models: XGBoost Classifier aur XGBoost Regressor (MLflow Model Registry mein registered).")
st.caption("Poora experiment tracking dashboard locally `mlflow ui` se dekha ja sakta hai.")
