# app.py — Page 1: Intro
import streamlit as st

# ---- minimal session-state bootstrap (no external files needed) ----
def init_state():
    defaults = {
        "df_raw": None,          # immutable copy of uploaded data
        "df_work": None,         # working copy that gets overwritten by cleaning/encoding
        "target_col": None,      # selected target column (e.g., Loan_Status)
        "id_cols": [],           # any ID columns to exclude from modeling
        "transforms": {          # audit of your data prep actions
            "imputations": {},   # {col: {"strategy": "mode|median", "filled": int}}
            "encodings": {}      # {col: {"type": "onehot|label", "details": {}}}
        },
        "train_report": None,    # metrics & params saved after training
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---- Streamlit page config ----
st.set_page_config(
    page_title="Loan Eligibility – Logistic Regression Workbench",
    page_icon="📘",
    layout="wide",
    menu_items={
        "Report a bug": None,
        "About": "A minimal end-to-end app to build, export, and use a Logistic Regression model for loan eligibility."
    }
)

init_state()

# ---- Main content ----
st.title("📘 Loan Eligibility – Logistic Regression Workbench")
st.caption("CSV/Excel → Clean/Encode → Visualize → Train Logistic Regression → Export → Predict")

st.markdown(
    """
**Welcome!** This app helps you build and ship a **Logistic Regression** model for loan eligibility:

1. **Load Data** — Upload CSV/XLSX, pick target & ID columns.  
2. **Clean & Encode** — Mode for categoricals, Median for numerics; One-Hot & Label encoding (column-wise), all applied **in-place**.  
3. **Visualize** — Quick EDA to understand distributions & relationships.  
4. **Train** — Fit a robust sklearn **Pipeline** that reproduces your preprocessing automatically.  
5. **Export** — Download `model.pkl` and `features.json`.  
6. **Predict** — Use `predict_proba` to report **Yes/No** with the higher probability.
"""
)

with st.expander("How does state work?", expanded=False):
    st.markdown(
        """
We keep two copies of your data:

- `df_raw`: immutable backup of the original upload.  
- `df_work`: the working DataFrame that gets **overwritten** by your cleaning/encoding actions.

Other keys:

- `target_col`, `id_cols`  
- `transforms`: audit of all imputations & encodings you applied  
- `train_report`: metrics & params from training
        """
    )

st.info("No data leaves your browser/server session. All processing is done within this app run.", icon="🔒")

st.subheader("Next step")
st.markdown("Go to **Load Data** from the sidebar or click the button below.")

# This will work once we create "pages/2_Load_Data.py".
if st.button("➡️ Go to Load Data"):
    try:
        st.switch_page("pages/2_Load_Data.py")
    except Exception:
        st.warning("The Load Data page isn't created yet. Use the sidebar once we add **pages/2_Load_Data.py**.")

st.markdown("---")
st.markdown("**Tip:** Keep your target column (e.g., `Loan_Status`) handy; we'll pick it on the next page.")

# Optional: gentle nudge in the sidebar
with st.sidebar:
    st.markdown("### Navigation")
    st.write("This is Page 1 (Intro). We’ll add **2_Load_Data** next.")
