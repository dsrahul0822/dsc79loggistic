# pages/7_Predict.py — Predict with raw features (categoricals allowed; auto-OHE inside pipeline)
import io
import json
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
init_state()

st.title("🔮 Predict Loan Eligibility")
st.caption("Upload **model.pkl** (and optional **features.json**). Enter RAW features — numeric and categorical — we’ll One-Hot them inside the pipeline.")

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def load_pickle_bytes(b: bytes):
    return pickle.loads(b)

def validate_and_align_columns(df: pd.DataFrame, required_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure df contains all required raw columns; fill missing as 0 for numeric and '' for categorical; drop extras."""
    missing = [c for c in required_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in required_cols]
    aligned = df.copy()
    for c in missing:
        if c in cat_cols:
            aligned[c] = ""
        else:
            aligned[c] = 0
    aligned = aligned[required_cols]
    return aligned, extra

def make_blank_template(required_cols: List[str], cat_cols: List[str]) -> bytes:
    if not required_cols:
        return b""
    row = []
    for c in required_cols:
        row.append("" if c in cat_cols else 0)
    df = pd.DataFrame([row], columns=required_cols)
    return df.to_csv(index=False).encode("utf-8")

# ---------------- Upload model & metadata ----------------
with st.container(border=True):
    st.subheader("1) Upload Model & Metadata")
    up_model = st.file_uploader("Upload model.pkl", type=["pkl"], accept_multiple_files=False)
    up_feat = st.file_uploader("Upload features.json (optional)", type=["json"], accept_multiple_files=False)

    model_bundle = None
    features_meta = None

    if up_model is not None:
        try:
            model_bytes = up_model.read()
            model_bundle = load_pickle_bytes(model_bytes)
            if not isinstance(model_bundle, dict) or "pipeline" not in model_bundle or "metadata" not in model_bundle:
                st.error("The uploaded pickle is not in the expected format. Please use the `model.pkl` exported from this app.", icon="❌")
                model_bundle = None
        except Exception as e:
            st.error(f"Could not load pickle: {e}", icon="❌")
            model_bundle = None

    if up_feat is not None:
        try:
            features_meta = json.loads(up_feat.read().decode("utf-8"))
        except Exception as e:
            st.error(f"Could not parse features.json: {e}", icon="❌")
            features_meta = None

if model_bundle is None:
    st.info("Upload your **model.pkl** to continue.", icon="ℹ️")
    st.stop()

pipe = model_bundle["pipeline"]
meta = model_bundle.get("metadata", {})

trained_features: List[str] = meta.get("features", [])
pos_label = meta.get("pos_label", 1)
neg_label = meta.get("neg_label", 0)
raw_cat_cols: List[str] = meta.get("raw_categorical_features", [])
ohe_categories: Dict[str, List[str]] = meta.get("ohe_categories", {})

st.success("Model loaded successfully.")
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.write("**Positive label (eligible):**", f"`{pos_label}`")
    st.write("**Negative label (not eligible):**", f"`{neg_label}`")
with c2:
    st.write("**#Raw features expected:**", len(trained_features))
with c3:
    st.write("**Feature names:**")
    st.caption(", ".join(trained_features[:10]) + (" ..." if len(trained_features) > 10 else ""))

st.download_button(
    "⬇️ Download RAW feature template CSV",
    data=make_blank_template(trained_features, raw_cat_cols),
    file_name="features_template_raw.csv",
    mime="text/csv",
    help="A one-row CSV with required raw columns (categoricals empty strings, numerics zeros)."
)

st.divider()

# ---------------- Single record prediction (raw features) ----------------
st.subheader("2) Single Record Prediction (RAW features)")
st.caption("Enter values for each raw feature. Categorical features have dropdowns (if category list is known) or text boxes; numeric features use numbers.")

with st.form("__single_record_form__"):
    user_vals: Dict[str, object] = {}
    for feat in trained_features:
        if feat in raw_cat_cols:
            cats = ohe_categories.get(feat, [])
            if cats:
                user_vals[feat] = st.selectbox(f"{feat} (categorical)", options=[""] + cats, index=0)
            else:
                user_vals[feat] = st.text_input(f"{feat} (categorical)", value="")
        else:
            user_vals[feat] = st.number_input(f"{feat} (numeric)", value=0.0, step=1.0, format="%.6f")
    submitted = st.form_submit_button("Predict (single record)")

if submitted:
    X_single = pd.DataFrame([user_vals], columns=trained_features)
    try:
        clf = pipe.named_steps.get("clf", None)
        proba = pipe.predict_proba(X_single)[0]
        classes_ = clf.classes_ if clf is not None else np.array([0, 1])

        # class 1 index
        if 1 in classes_:
            idx_pos = int(np.where(classes_ == 1)[0][0])
        else:
            idx_pos = len(classes_) - 1

        p_yes = float(proba[idx_pos])
        p_no = float(1.0 - p_yes) if len(classes_) == 2 else float(proba[1 - idx_pos])

        if p_yes >= p_no:
            st.success(f"✅ This customer will **GET a loan** with probability **{p_yes:.3f}** (Yes).")
            st.caption(f"No probability: {p_no:.3f}")
        else:
            st.error(f"❌ This customer will **NOT get a loan** with probability **{p_no:.3f}** (No).")
            st.caption(f"Yes probability: {p_yes:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}", icon="❌")

st.divider()

# ---------------- Batch prediction ----------------
st.subheader("3) Batch Prediction (CSV of RAW features)")
st.caption("Upload a CSV with the same raw feature columns used during training. Missing columns are added ('' for categoricals, 0 for numerics). Extra columns are dropped.")

up_batch = st.file_uploader("Upload batch features CSV", type=["csv"], accept_multiple_files=False, key="__batch_csv__")

if up_batch is not None:
    try:
        df_in = pd.read_csv(up_batch)
        st.write(f"Uploaded shape: {df_in.shape[0]} rows × {df_in.shape[1]} columns")

        # Align columns
        aligned, extra = validate_and_align_columns(df_in, trained_features, raw_cat_cols)

        if extra:
            st.warning(f"Dropped {len(extra)} extra column(s): {extra[:10]}{' ...' if len(extra)>10 else ''}")

        # Predict
        proba = pipe.predict_proba(aligned)
        clf = pipe.named_steps.get("clf", None)
        classes_ = clf.classes_ if clf is not None else np.array([0, 1])

        if 1 in classes_:
            idx_pos = int(np.where(classes_ == 1)[0][0])
        else:
            idx_pos = len(classes_) - 1

        p_yes = proba[:, idx_pos]
        p_no = 1.0 - p_yes if proba.shape[1] == 2 else proba[:, 1 - idx_pos]
        pred_yes = (p_yes >= p_no)

        out = df_in.copy()
        out["proba_yes"] = p_yes
        out["proba_no"] = p_no
        out["prediction"] = np.where(pred_yes, str(pos_label), str(neg_label))
        out["decision_text"] = np.where(
            pred_yes,
            [f"GET loan (Yes) with p={p:.3f}" for p in p_yes],
            [f"NOT get loan (No) with p={p:.3f}" for p in p_no],
        )

        st.success("Batch predictions computed.")
        st.dataframe(out.head(20), use_container_width=True)

        st.download_button(
            "⬇️ Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Batch prediction failed: {e}", icon="❌")

st.divider()
st.subheader("Navigation")
col1, col2 = st.columns(2)
with col1:
    if st.button("⬅️ Back to Train Model"):
        try:
            st.switch_page("pages/5_Train_Model.py")
        except Exception:
            st.warning("Use the left sidebar → **5_Train_Model**.")
with col2:
    if st.button("⬅️ Back to Visuals"):
        try:
            st.switch_page("pages/4_Visuals.py")
        except Exception:
            st.warning("Use the left sidebar → **4_Visuals**.")
