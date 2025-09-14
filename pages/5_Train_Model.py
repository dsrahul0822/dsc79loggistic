# pages/5_Train_Model.py — Train Logistic Regression, auto-encode categoricals, persistent downloads
import io
import json
import pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- minimal session-state bootstrap (standalone friendly) ----
def init_state():
    defaults = {
        "df_raw": None,
        "df_work": None,
        "target_col": None,
        "id_cols": [],
        "transforms": {"imputations": {}, "encodings": {}},
        "train_report": None,
        # NEW: persist downloadable bytes across reruns
        "export_model_bytes": None,
        "export_features_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

st.set_page_config(page_title="Train Model", page_icon="🧠", layout="wide")
init_state()

st.title("🧠 Train Logistic Regression")
st.caption("Robust to raw categoricals (auto One-Hot with handle_unknown='ignore'). Downloads persist after clicks.")

if st.session_state.df_work is None:
    st.info("No dataset loaded yet. Go to **Load Data** and click “Use this as my dataset”.", icon="ℹ️")
    st.stop()

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def check_no_nans(df: pd.DataFrame, cols: List[str]) -> Tuple[bool, pd.Series]:
    miss = df[cols].isna().sum()
    has_nan = (miss > 0).any()
    return (not has_nan), miss[miss > 0]

def split_numeric_columns(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    num_cols, non_num_cols = [], []
    for c in features:
        if is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            non_num_cols.append(c)
    return num_cols, non_num_cols

def split_float_int_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[List[str], List[str]]:
    floats, ints = [], []
    for c in numeric_cols:
        dt = str(df[c].dtype)
        (floats if dt.startswith("float") else ints).append(c)
    return floats, ints

def make_preprocessor(X: pd.DataFrame, features: List[str], scale_numeric: bool, floats_only: bool
) -> Tuple[ColumnTransformer, Dict[str, List[str]]]:
    num_cols, cat_cols = split_numeric_columns(X, features)
    float_nums, int_nums = split_float_int_numeric(X, num_cols)

    transformers = []
    scale_cols_record = []

    if scale_numeric:
        if floats_only:
            if float_nums:
                transformers.append(("scale", StandardScaler(), float_nums))
                scale_cols_record.extend(float_nums)
            if int_nums:
                transformers.append(("num_pass", "passthrough", int_nums))
        else:
            if num_cols:
                transformers.append(("scale", StandardScaler(), num_cols))
                scale_cols_record.extend(num_cols)
    else:
        if num_cols:
            transformers.append(("num_pass", "passthrough", num_cols))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    schema = {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "float_numeric_scaled": scale_cols_record,
    }
    return pre, schema

def pack_model_bytes(model) -> bytes:
    return pickle.dumps(model)

def json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")

def show_download_buttons():
    """Always show if bytes exist in session; survives reruns and multiple clicks."""
    mb = st.session_state.get("export_model_bytes", None)
    fb = st.session_state.get("export_features_bytes", None)

    st.markdown("### 5) Export Artifacts")
    if mb is None and fb is None:
        st.info("Train the model to enable downloads.", icon="ℹ️")
        return

    c1, c2 = st.columns(2)
    with c1:
        if mb is not None:
            st.download_button(
                "⬇️ Download model.pkl",
                data=mb,
                file_name="model.pkl",
                mime="application/octet-stream",
                help="Pickle of a dict with keys: pipeline (sklearn Pipeline) + metadata."
            )
        else:
            st.button("⬇️ Download model.pkl", disabled=True)

    with c2:
        if fb is not None:
            st.download_button(
                "⬇️ Download features.json",
                data=fb,
                file_name="features.json",
                mime="application/json",
                help="Feature schema and categorical choices for the Predict page."
            )
        else:
            st.button("⬇️ Download features.json", disabled=True)

# ----------------------------------------------------------------------
# Data + UI
# ----------------------------------------------------------------------
dfw = st.session_state.df_work.copy()
all_cols = dfw.columns.tolist()
target_col = st.session_state.target_col
id_cols = st.session_state.id_cols or []

st.subheader("1) Target & Features")

if target_col not in all_cols:
    target_col = st.selectbox("Select target column", options=all_cols, index=0 if len(all_cols) else None, key="__target_pick__")
else:
    st.markdown(f"**Target column:** `{target_col}`")

default_feats = [c for c in all_cols if c != target_col and c not in id_cols]
selected_features = st.multiselect(
    "Select feature columns for training",
    options=[c for c in all_cols if c != target_col],
    default=default_feats,
)
if len(selected_features) == 0:
    st.warning("Please select at least one feature.", icon="⚠️")
    show_download_buttons()
    st.stop()

st.write(f"Training frame shape if used now: **{dfw.shape[0]} rows × {len(selected_features)+1} columns (features + target)**")

# Binary target mapping
st.subheader("2) Target Mapping (Binary)")
unique_targets = pd.Series(dfw[target_col].dropna().unique()).tolist()
unique_targets_preview = ", ".join([str(u) for u in unique_targets][:6])
st.caption(f"Unique values in `{target_col}` (showing up to 6): {unique_targets_preview}")
if len(unique_targets) < 2:
    st.error("Target must have at least two classes.", icon="❌")
    show_download_buttons()
    st.stop()

if len(unique_targets) > 2:
    st.warning("Detected more than 2 unique target values. Select the two classes to use (others will be filtered).")
    pos_label = st.selectbox("Positive class label (Yes/Eligible)", options=unique_targets, key="__pos_lbl__")
    neg_options = [u for u in unique_targets if u != pos_label]
    neg_label = st.selectbox("Negative class label (No/Not Eligible)", options=neg_options, key="__neg_lbl__")
    dfw = dfw[dfw[target_col].isin([pos_label, neg_label])].copy()
else:
    heur_pos = None
    lower_set = {str(x).strip().lower() for x in unique_targets}
    if "y" in lower_set: heur_pos = [x for x in unique_targets if str(x).strip().lower()=="y"][0]
    if heur_pos is None and "yes" in lower_set: heur_pos = [x for x in unique_targets if str(x).strip().lower()=="yes"][0]
    pos_label = st.selectbox("Positive class label (Yes/Eligible)", options=unique_targets, index=(unique_targets.index(heur_pos) if heur_pos in unique_targets else 0))
    neg_label = [u for u in unique_targets if u != pos_label][0]

st.info(f"Binary mapping: **{pos_label} → 1**, **{neg_label} → 0**")

y = dfw[target_col].map(lambda v: 1 if v == pos_label else (0 if v == neg_label else np.nan)).astype("float")
nan_y = int(y.isna().sum())
if nan_y > 0:
    st.warning(f"Dropping {nan_y} rows where target is neither `{pos_label}` nor `{neg_label}`.")
    mask = y.notna()
    dfw = dfw.loc[mask].copy()
    y = y.loc[mask].astype("int")

X = dfw[selected_features].copy()

# Validate missing values in features
ok_no_nan, missing_series = check_no_nans(X, selected_features)
if not ok_no_nan:
    st.error("Some selected features still contain missing values. Please impute them on **Clean & Encode** before training.", icon="❌")
    st.dataframe(missing_series.to_frame("missing_count"))
    show_download_buttons()
    st.stop()

# ----------------------------------------------------------------------
st.subheader("3) Training Options")

c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Test size (fraction)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
with c2:
    scale_numeric = st.checkbox("Standardize numeric features", value=True)
    floats_only = st.checkbox("Scale only float columns (leave ints/dummies as-is)", value=True)
with c3:
    class_bal = st.checkbox("Class weight = balanced", value=False)

c4, c5, c6 = st.columns(3)
with c4:
    solver = st.selectbox("Solver", options=["liblinear", "lbfgs", "saga"], index=0)
with c5:
    C = st.number_input("Inverse regularization strength (C)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
with c6:
    max_iter = st.number_input("Max iterations", min_value=100, max_value=10000, value=500, step=100)

# Build preprocessing that auto-encodes non-numeric columns
pre, schema = make_preprocessor(X, selected_features, scale_numeric=scale_numeric, floats_only=floats_only)
clf = LogisticRegression(
    C=C,
    class_weight=("balanced" if class_bal else None),
    solver=solver,
    max_iter=int(max_iter),
)
pipe = Pipeline([("prep", pre), ("clf", clf)])

# ----------------------------------------------------------------------
st.subheader("4) Train")
if st.button("🚀 Train Logistic Regression"):
    with st.spinner("Training..."):
        # Optional: clear previous bytes so user doesn't accidentally download old model
        st.session_state.export_model_bytes = None
        st.session_state.export_features_bytes = None

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Fit
        pipe.fit(X_train, y_train)

        # Predict & metrics
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]  # probability of class 1 (pos_label)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        # Collect OHE categories (if any)
        ohe_categories = {}
        try:
            cat_enc = pipe.named_steps["prep"].named_transformers_.get("cat")
            cat_cols = schema.get("categorical_features", [])
            if cat_enc is not None and hasattr(cat_enc, "categories_"):
                for col, cats in zip(cat_cols, cat_enc.categories_):
                    ohe_categories[col] = [str(c) for c in cats.tolist()]
        except Exception:
            pass

        # Save training report
        st.session_state.train_report = {
            "accuracy": float(acc),
            "test_size": float(test_size),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "target_col": target_col,
            "pos_label": pos_label,
            "neg_label": neg_label,
            "features": selected_features,                           # raw features expected at predict time
            "raw_numeric_features": schema.get("numeric_features", []),
            "raw_categorical_features": schema.get("categorical_features", []),
            "scaled_numeric_features": schema.get("float_numeric_scaled", []),
            "class_weight": ("balanced" if class_bal else "none"),
            "solver": solver,
            "C": float(C),
            "max_iter": int(max_iter),
            "ohe_categories": ohe_categories,
            "transforms_snapshot": st.session_state.transforms,
        }

        # ---- Build export bundle and persist BYTES in session ----
        export_bundle = {
            "pipeline": pipe,
            "metadata": {
                "features": selected_features,
                "target_col": target_col,
                "pos_label": pos_label,
                "neg_label": neg_label,
                "id_cols": id_cols,
                "scale_numeric": bool(scale_numeric),
                "floats_only": bool(floats_only),
                "class_weight": ("balanced" if class_bal else "none"),
                "solver": solver,
                "C": float(C),
                "max_iter": int(max_iter),
                "raw_numeric_features": schema.get("numeric_features", []),
                "raw_categorical_features": schema.get("categorical_features", []),
                "scaled_numeric_features": schema.get("float_numeric_scaled", []),
                "ohe_categories": ohe_categories,
                "transforms_snapshot": st.session_state.transforms,
            }
        }
        st.session_state.export_model_bytes = pack_model_bytes(export_bundle)

        features_json = {
            "features": export_bundle["metadata"]["features"],
            "target_col": target_col,
            "pos_label": pos_label,
            "neg_label": neg_label,
            "id_cols": id_cols,
            "raw_numeric_features": export_bundle["metadata"]["raw_numeric_features"],
            "raw_categorical_features": export_bundle["metadata"]["raw_categorical_features"],
            "scaled_numeric_features": export_bundle["metadata"]["scaled_numeric_features"],
            "ohe_categories": export_bundle["metadata"]["ohe_categories"],
        }
        st.session_state.export_features_bytes = json_bytes(features_json)

        # ---- Display metrics ----
        st.success(f"Model trained. Accuracy: **{acc:.4f}**  (n_test={X_test.shape[0]})")

        colA, colB = st.columns([1, 1.2])
        with colA:
            st.markdown("**Confusion Matrix**")
            fig, ax = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                        xticklabels=[f"Pred {neg_label}", f"Pred {pos_label}"],
                        yticklabels=[f"True {neg_label}", f"True {pos_label}"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with colB:
            st.markdown("**Classification Report**")
            rep = classification_report(y_test, y_pred, target_names=[str(neg_label), str(pos_label)], output_dict=False)
            st.text(rep)

# Always show downloads if bytes exist (survives reruns & clicks)
st.markdown("---")
show_download_buttons()

st.divider()
st.subheader("Next step")
st.markdown("Go to **Predict** to use your model.")
if st.button("➡️ Proceed to Predict"):
    try:
        st.switch_page("pages/7_Predict.py")
    except Exception:
        st.warning("Use the left sidebar → **7_Predict** to continue.")
