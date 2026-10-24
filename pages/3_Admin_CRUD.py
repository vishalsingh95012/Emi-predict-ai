import streamlit as st, pandas as pd, os, shutil
st.title("🛠️ Admin: Data Management (CRUD)")

PATH = "data/records.csv"
if not os.path.exists(PATH):
    shutil.copy("data/sample.csv", PATH)

if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(PATH)

st.caption("Edit: cell par double click. Add: table ke neeche '+' row. Delete: row select karke Delete key. Phir Save dabao.")
edited = st.data_editor(st.session_state.df, num_rows="dynamic", height=450)

c1, c2, c3 = st.columns([1, 1, 1])
if c1.button("💾 Save changes", type="primary"):
    st.session_state.df = edited.reset_index(drop=True)
    st.session_state.df.to_csv(PATH, index=False)
    st.success(f"Saved. Total records: {len(st.session_state.df):,}")
if c2.button("↩️ Reset to original"):
    shutil.copy("data/sample.csv", PATH)
    st.session_state.df = pd.read_csv(PATH)
    st.rerun()
c3.download_button("⬇️ Download CSV", st.session_state.df.to_csv(index=False), "records.csv")
st.write(f"Total records: {len(st.session_state.df):,}")
