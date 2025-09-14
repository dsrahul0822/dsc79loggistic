# pages/3_Clean_Encode.py — Page 3: Missing Value Treatment + Encoding (in-place)
import io
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype, is_string_dtype

# ---- minimal session-state bootstrap (standalone friendly) ----
def init_state():
    defaults = {
        "df_raw": None,          # immutable copy of uploaded data
        "df_work": None,         # working copy (we overwrite this)
        "target_col": None,      # selected target column (e.g., Loan_Status)
        "id_cols": [],           # any ID columns to exclude from modeling
        "transforms": {          # audit of data prep actions (used later to build pipeline)
            "imputations": {},   # {col: {"strategy": "mode|median", "filled": int, "value": any}}
            "encodings": {}      # {col: {"type": "onehot|label", "details": {...}}}
        },
        "train_report": None,    # metrics & params after training
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(page_title="Clean & Encode", page_icon="🧼", layout="wide")
init_state()

st.title("🧼 Clean & Encode")
st.caption("Apply **missing value treatment** (Mode/Median) and **encoding** (One-Hot/Label) to the working dataset. All actions overwrite `df_work`.")

if st.session_state.df_work is None:
    st.info("No dataset loaded yet. Go to **Load Data** first and click “Use this as my dataset”.", icon="ℹ️")
    st.stop()

dfw = st.session_state.df_work

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def list_categorical_columns(df: pd.DataFrame):
    cats = []
    for c in df.columns:
        if is_bool_dtype(df[c]) or is_categorical_dtype(df[c]) or is_string_dtype(df[c]) or df[c].dtype == "object":
            cats.append(c)
    return cats

def list_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]

def mode_value(series: pd.Series):
    m = series.mode(dropna=True)
    if len(m) == 0:
        return None
    return m.iloc[0]

def label_encode_series(series: pd.Series):
    """Deterministic mapping: sorted unique (as strings) -> integers."""
    # Convert to string to avoid mixed types
    vals = series.astype("string")
    uniques = sorted([u for u in vals.dropna().unique().tolist()])
    mapping = {u: i for i, u in enumerate(uniques)}
    encoded = vals.map(lambda x: mapping.get(x) if pd.notna(x) else np.nan)
    return encoded.astype("Int64"), mapping

def one_hot_encode_column(df: pd.DataFrame, col: str, drop_original=True, prefix=None):
    if prefix is None:
        prefix = col
    dummies = pd.get_dummies(df[col], prefix=prefix, dummy_na=False)
    new_df = pd.concat([df.drop(columns=[col]) if drop_original else df, dummies], axis=1)
    return new_df, dummies.columns.tolist()

def mark_imputation(col: str, strategy: str, filled: int, value):
    st.session_state.transforms["imputations"][col] = {
        "strategy": strategy,
        "filled": int(filled),
        "value": value if (isinstance(value, (int, float, str, bool)) or value is None) else str(value)
    }

def mark_encoding(col: str, enc_type: str, details: dict):
    st.session_state.transforms["encodings"][col] = {
        "type": enc_type,
        "details": details
    }

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.subheader("1) Missing Value Treatment (Column-wise)")
    st.write("Choose a column and apply the appropriate strategy. This will **overwrite** `df_work` in place.")

    cols = dfw.columns.tolist()
    target_col = st.session_state.target_col
    id_cols = st.session_state.id_cols or []

    # Column picker for imputation (including target if needed; warn separately)
    imp_col = st.selectbox("Select a column to impute", options=cols, index=0 if len(cols) else None, key="__imp_col__")

    # Show quick stats for the selected column
    if imp_col:
        missing_count = int(dfw[imp_col].isna().sum())
        st.caption(f"`{imp_col}` – dtype: **{dfw[imp_col].dtype}**, missing: **{missing_count}**")

        # Strategy options (fixed per dtype as per your spec)
        if is_numeric_dtype(dfw[imp_col]) and not is_bool_dtype(dfw[imp_col]):
            st.write("Strategy: **Median** (numeric)")
            if st.button("Apply Median Imputation to this column"):
                med = dfw[imp_col].median(skipna=True)
                if pd.isna(med):
                    st.warning("Median could not be computed (all values NaN?). No changes made.")
                else:
                    filled_before = dfw[imp_col].isna().sum()
                    dfw[imp_col] = dfw[imp_col].fillna(med)
                    filled_after = dfw[imp_col].isna().sum()
                    mark_imputation(imp_col, "median", filled_before - filled_after, med)
                    st.session_state.df_work = dfw
                    st.success(f"Applied Median={med} to `{imp_col}`. Missing filled: {int(filled_before - filled_after)}")
        else:
            st.write("Strategy: **Mode** (categorical)")
            if st.button("Apply Mode Imputation to this column"):
                mv = mode_value(dfw[imp_col])
                if mv is None:
                    st.warning("Mode could not be computed (all values NaN?). No changes made.")
                else:
                    filled_before = dfw[imp_col].isna().sum()
                    dfw[imp_col] = dfw[imp_col].fillna(mv)
                    filled_after = dfw[imp_col].isna().sum()
                    mark_imputation(imp_col, "mode", filled_before - filled_after, mv)
                    st.session_state.df_work = dfw
                    st.success(f"Applied Mode='{mv}' to `{imp_col}`. Missing filled: {int(filled_before - filled_after)}")

    with st.expander("⚡ Auto-impute defaults for all columns", expanded=False):
        c1, c2 = st.columns(2)
        if c1.button("Auto: Mode for all categorical"):
            cats = [c for c in list_categorical_columns(dfw)]
            total_filled = 0
            for c in cats:
                mv = mode_value(dfw[c])
                if mv is not None:
                    before = dfw[c].isna().sum()
                    dfw[c] = dfw[c].fillna(mv)
                    after = dfw[c].isna().sum()
                    delta = int(before - after)
                    total_filled += delta
                    mark_imputation(c, "mode", delta, mv)
            st.session_state.df_work = dfw
            st.success(f"Auto-imputed categorical columns with Mode. Total values filled: {total_filled}")

        if c2.button("Auto: Median for all numeric"):
            nums = list_numeric_columns(dfw)
            total_filled = 0
            for c in nums:
                med = dfw[c].median(skipna=True)
                if not pd.isna(med):
                    before = dfw[c].isna().sum()
                    dfw[c] = dfw[c].fillna(med)
                    after = dfw[c].isna().sum()
                    delta = int(before - after)
                    total_filled += delta
                    mark_imputation(c, "median", delta, med)
            st.session_state.df_work = dfw
            st.success(f"Auto-imputed numeric columns with Median. Total values filled: {total_filled}")

    st.divider()
    st.subheader("2) Encoding (Column-wise)")
    st.write("Pick a feature, choose **One-Hot** (categorical) or **Label Encoding** (e.g., `Dependents`). The operation will **overwrite** `df_work`.")

    # Eligible columns (avoid target; usually you don't encode IDs either)
    eligible_cols = [c for c in dfw.columns if c != target_col and c not in (id_cols or [])]

    enc_col = st.selectbox("Select a column to encode", options=eligible_cols, index=0 if len(eligible_cols) else None, key="__enc_col__")
    enc_type = st.radio("Encoding type", options=["One-Hot Encoding", "Label Encoding"], horizontal=True)

    if enc_col:
        if enc_col == target_col:
            st.error("Target column should not be encoded here. We handle target encoding during training.")
        elif enc_type == "One-Hot Encoding":
            if is_numeric_dtype(dfw[enc_col]) and not is_bool_dtype(dfw[enc_col]):
                st.warning(f"`{enc_col}` is numeric; One-Hot is typically for categoricals. Proceed only if intended.")
            if st.button("Apply One-Hot Encoding"):
                new_df, new_cols = one_hot_encode_column(dfw, enc_col, drop_original=True, prefix=enc_col)
                st.session_state.df_work = new_df
                mark_encoding(enc_col, "onehot", {"created_columns": new_cols})
                st.success(f"One-Hot encoded `{enc_col}` → created {len(new_cols)} columns. Original column removed.")
        else:  # Label Encoding
            if is_numeric_dtype(dfw[enc_col]) and not is_bool_dtype(dfw[enc_col]):
                st.warning(f"`{enc_col}` looks numeric already. Label Encoding is for categorical strings.")
            if st.button("Apply Label Encoding"):
                encoded, mapping = label_encode_series(dfw[enc_col])
                before_nans = int(encoded.isna().sum())
                # If there were NaNs, fill with a special code (e.g., -1)
                encoded = encoded.fillna(-1).astype("int64")
                st.session_state.df_work[enc_col] = encoded
                mark_encoding(enc_col, "label", {"mapping": mapping, "nan_filled_to": -1, "nan_count": before_nans})
                st.success(f"Label-encoded `{enc_col}` with {len(mapping)} categories. NaNs mapped to -1 (count={before_nans}).")

with right:
    st.subheader("Working Data Snapshot")
    st.caption("This is what will be used for visuals and training.")
    st.dataframe(st.session_state.df_work.head(20), use_container_width=True)

    st.subheader("Missing Values (current)")
    miss = st.session_state.df_work.isna().sum().sort_values(ascending=False)
    st.dataframe(miss.to_frame("missing_count"), use_container_width=True)

    with st.expander("Transforms audit", expanded=False):
        st.json(st.session_state.transforms, expanded=False)

st.divider()
st.subheader("Next step")
st.markdown("Go to **Visuals** from the sidebar or click the button below.")
if st.button("➡️ Proceed to Visuals"):
    try:
        st.switch_page("pages/4_Visuals.py")
    except Exception:
        st.warning("Use the left sidebar → **4_Visuals** to continue.")
