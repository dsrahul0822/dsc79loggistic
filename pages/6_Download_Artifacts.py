# pages/6_Download_Artifacts.py — Helper summary for artifacts
import streamlit as st
import pandas as pd

def init_state():
    defaults = {
        "df_raw": None,
        "df_work": None,
        "target_col": None,
        "id_cols": [],
        "transforms": {"imputations": {}, "encodings": {}},
        "train_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(page_title="Download Artifacts", page_icon="📦", layout="wide")
init_state()

st.title("📦 Download Artifacts")
st.caption("Model and feature metadata are downloaded from the **Train Model** page after training.")

if st.session_state.train_report is None:
    st.info("No training summary found in this session. Train your model first on **5_Train_Model**.", icon="ℹ️")
else:
    st.success("Training summary found. Use the buttons on **5_Train_Model** to download `model.pkl` and `features.json`.")
    st.subheader("Training Summary (snapshot)")
    st.json(st.session_state.train_report, expanded=False)

st.divider()
st.subheader("Next")
st.markdown("- Go to **5_Train_Model** to re-train or download artifacts.")
if st.button("⬅️ Back to Train Model"):
    try:
        st.switch_page("pages/5_Train_Model.py")
    except Exception:
        st.warning("Use the left sidebar → **5_Train_Model**.")
