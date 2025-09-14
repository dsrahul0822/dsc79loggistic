# pages/2_Load_Data.py — Page 2: Load Data (CSV/XLSX upload + preview + target/ID selection)
import io
import pandas as pd
import streamlit as st

# ---- minimal session-state bootstrap (kept local so this page works standalone) ----
def init_state():
    defaults = {
        "df_raw": None,          # immutable copy of uploaded data
        "df_work": None,         # working copy that gets overwritten by cleaning/encoding
        "target_col": None,      # selected target column (e.g., Loan_Status)
        "id_cols": [],           # any ID columns to exclude from modeling
        "transforms": {          # audit of data prep actions (used later to build pipeline)
            "imputations": {},   # {col: {"strategy": "mode|median", "filled": int}}
            "encodings": {}      # {col: {"type": "onehot|label", "details": {}}}
        },
        "train_report": None,    # metrics & params after training
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(page_title="Load Data", page_icon="📥", layout="wide")
init_state()

st.title("📥 Load Data")
st.caption("Upload a CSV/XLSX, preview it, and set Target & ID columns. We’ll use this working copy on the next pages.")

# -------- Helpers --------
def _suggest_target_column(columns):
    """Heuristic: try common target names."""
    candidates = ["Loan_Status", "loan_status", "TARGET", "target", "label", "Label", "y"]
    for c in candidates:
        if c in columns:
            return c
    return None

def _suggest_id_columns(columns):
    """Heuristic: pick obvious ID-like columns."""
    id_like = []
    for c in columns:
        lc = c.lower()
        if "id" == lc or lc.endswith("_id") or "loan_id" in lc or lc in ("id", "customer_id", "applicant_id"):
            id_like.append(c)
    return id_like

def _read_csv(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
        return df

def _read_xlsx(uploaded_file, sheet_name=None) -> pd.DataFrame:
    # Read into memory once so we can reuse
    data = uploaded_file.read()
    bio = io.BytesIO(data)
    xls = pd.ExcelFile(bio)
    if sheet_name is None:
        sheet_name = xls.sheet_names[0]
    bio2 = io.BytesIO(data)
    df = pd.read_excel(bio2, sheet_name=sheet_name)
    return df, xls.sheet_names

def _finalize_loaded_df(df: pd.DataFrame):
    # Light normalization: convert dtypes & strip column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.convert_dtypes()
    return df

# -------- UI: Uploader --------
with st.container(border=True):
    st.subheader("1) Upload your dataset")
    up = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], accept_multiple_files=False)

    read_ok = False
    df_preview = None
    sheet_names = None

    if up is not None:
        if up.name.lower().endswith(".csv"):
            with st.spinner("Reading CSV..."):
                df_preview = _finalize_loaded_df(_read_csv(up))
                read_ok = True
        elif up.name.lower().endswith(".xlsx"):
            with st.spinner("Reading Excel..."):
                # First pass: get sheet names
                up.seek(0)
                tmp_df, sheet_names = _read_xlsx(up, None)
            # Sheet selector UI
            st.selectbox("Choose sheet", options=sheet_names, key="__load_sheet_name__")
            if st.button("Read selected sheet"):
                up.seek(0)
                with st.spinner(f"Reading sheet: {st.session_state['__load_sheet_name__']}"):
                    df_preview, _ = _read_xlsx(up, st.session_state["__load_sheet_name__"])
                    df_preview = _finalize_loaded_df(df_preview)
                    read_ok = True

    if read_ok and df_preview is not None:
        st.success(f"Loaded shape: {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df_preview.head(10), use_container_width=True)
        with st.expander("Detected dtypes", expanded=False):
            dtypes_df = pd.DataFrame({"column": df_preview.columns, "dtype": df_preview.dtypes.astype(str)})
            st.dataframe(dtypes_df, use_container_width=True)

        # Save to session
        if st.button("✅ Use this as my dataset"):
            st.session_state.df_raw = df_preview.copy()
            st.session_state.df_work = df_preview.copy()
            st.session_state.transforms = {"imputations": {}, "encodings": {}}
            st.session_state.train_report = None
            # Suggest target & ids
            suggested_target = _suggest_target_column(df_preview.columns)
            st.session_state.target_col = suggested_target
            st.session_state.id_cols = _suggest_id_columns(df_preview.columns)
            st.success("Dataset stored in session as df_raw (backup) and df_work (working copy). Proceed to step 2.")

# -------- UI: Target & IDs --------
if st.session_state.df_work is not None:
    st.subheader("2) Target & ID columns")
    cols = list(st.session_state.df_work.columns)

    # Target selection
    target_default = 0
    suggested = _suggest_target_column(cols)
    if suggested and suggested in cols:
        target_default = cols.index(suggested)

    target = st.selectbox("Select target column (e.g., Loan_Status)", options=cols, index=target_default if len(cols) else 0, key="__target_picker__")

    # ID columns
    suggested_ids = _suggest_id_columns(cols)
    id_cols = st.multiselect("Select ID columns to exclude from modeling (kept for reference)", options=cols, default=[c for c in suggested_ids if c in cols], key="__id_picker__")

    c1, c2, c3 = st.columns([1,1,2], vertical_alignment="center")
    with c1:
        if st.button("💾 Save selections"):
            st.session_state.target_col = target
            st.session_state.id_cols = id_cols
            st.success(f"Saved. Target: `{st.session_state.target_col}`, IDs: {st.session_state.id_cols}")
    with c2:
        if st.button("🔄 Reset working copy to raw"):
            if st.session_state.df_raw is not None:
                st.session_state.df_work = st.session_state.df_raw.copy()
                st.session_state.transforms = {"imputations": {}, "encodings": {}}
                st.success("df_work has been reset to df_raw. All previous cleaning/encoding logs cleared.")
            else:
                st.warning("No df_raw found. Upload a dataset first.")

    # Simple health snapshot
    with st.expander("Data health snapshot", expanded=False):
        dfw = st.session_state.df_work
        st.write(f"Rows: {dfw.shape[0]}, Columns: {dfw.shape[1]}")
        missing = dfw.isna().sum().sort_values(ascending=False)
        st.write("Top missing-value columns:")
        st.dataframe(missing.to_frame("missing_count"), use_container_width=True)

    st.divider()
    st.subheader("Next step")
    st.markdown("Go to **Clean & Encode** from the sidebar or click the button below.")
    if st.button("➡️ Proceed to Clean & Encode"):
        try:
            st.switch_page("pages/3_Clean_Encode.py")
        except Exception:
            st.warning("Use the left sidebar → **3_Clean_Encode** to continue.")

else:
    st.info("Upload a file above and click **Use this as my dataset** to enable Target/ID selection.", icon="ℹ️")
