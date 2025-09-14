# pages/4_Visuals.py — Page 4: Quick Visuals
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype, is_string_dtype

# ---- minimal session-state bootstrap (standalone friendly) ----
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

st.set_page_config(page_title="Visuals", page_icon="📊", layout="wide")
init_state()

st.title("📊 Quick Visuals")
st.caption("Fast EDA on the **working** dataset (`df_work`). Use this to sanity-check distributions and relationships before training.")

if st.session_state.df_work is None:
    st.info("No dataset loaded yet. Go to **Load Data** and click “Use this as my dataset”.", icon="ℹ️")
    st.stop()

dfw = st.session_state.df_work.copy()
target_col = st.session_state.target_col
id_cols = st.session_state.id_cols or []

# ---------- helpers ----------
def list_categorical_columns(df: pd.DataFrame):
    cats = []
    for c in df.columns:
        if is_bool_dtype(df[c]) or is_categorical_dtype(df[c]) or is_string_dtype(df[c]) or df[c].dtype == "object":
            cats.append(c)
    return cats

def list_numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]

def grid(n_items, max_cols=3):
    cols = min(max_cols, max(1, n_items))
    rows = math.ceil(n_items / cols)
    return rows, cols

sns.set_theme(style="whitegrid")

# ---------- sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    st.write("Configure what to plot below.")

    show_missing_bar = st.checkbox("Show Top Missing-Value Columns", value=True)

    st.markdown("---")
    st.subheader("Categorical Count Plots")
    all_cats = [c for c in list_categorical_columns(dfw) if c not in id_cols]
    cats_sel = st.multiselect("Pick categorical columns", options=all_cats, default=all_cats[:4])
    topK = st.slider("Top K categories per plot (by frequency)", 3, 30, 12)

    st.markdown("---")
    st.subheader("Numeric Histograms")
    all_nums = [c for c in list_numeric_columns(dfw) if c not in id_cols]
    nums_sel = st.multiselect("Pick numeric columns", options=all_nums, default=all_nums[:4])
    bins = st.slider("Bins", 5, 100, 30)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    drop_ids = st.checkbox("Exclude ID columns from heatmap", value=True)

    st.markdown("---")
    st.subheader("Target vs Feature")
    hue_ok = target_col in dfw.columns if target_col else False
    st.caption(f"Target column: **{target_col or 'not set'}**")
    feat_vs_target = st.selectbox("Pick a feature to compare with target", options=[c for c in dfw.columns if c != target_col], index=0 if len(dfw.columns)>0 else None)
    kind = st.radio("Plot type", ["Auto", "Bar (cat)", "Box (num)"], horizontal=True)

# ---------- missing bar ----------
if show_missing_bar:
    st.subheader("Missing-Value Snapshot")
    miss = dfw.isna().sum().sort_values(ascending=False)
    miss = miss[miss>0]
    if miss.empty:
        st.success("No missing values found in the current working dataset 🎉")
    else:
        fig, ax = plt.subplots(figsize=(8, max(3, len(miss)*0.4)))
        miss.head(25).plot(kind="barh", ax=ax)
        ax.set_xlabel("Missing Count")
        ax.set_ylabel("Column")
        ax.set_title("Top Missing-Value Columns (up to 25)")
        st.pyplot(fig)
    st.divider()

# ---------- categorical count plots ----------
if cats_sel:
    st.subheader("Count Plots (Categorical)")
    rows, cols = grid(len(cats_sel), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for i, col in enumerate(cats_sel):
        ax = axes[i]
        vc = dfw[col].fillna("NA").astype(str).value_counts().head(topK)
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_title(f"{col} (Top {min(topK, vc.shape[0])})")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        if target_col in dfw.columns:
            ax.text(0.98, 0.05, f"Target: {target_col}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="#555")
    # hide empty axes
    for j in range(i+1, rows*cols):
        axes[j].axis("off")
    st.pyplot(fig)
    st.divider()

# ---------- numeric histograms ----------
if nums_sel:
    st.subheader("Histograms (Numeric)")
    rows, cols = grid(len(nums_sel), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows))
    axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])

    for i, col in enumerate(nums_sel):
        ax = axes[i]
        data = dfw[col].dropna()
        if data.nunique() <= 1:
            ax.text(0.5, 0.5, f"{col}\n(constant)", ha="center", va="center")
            ax.axis("off")
            continue
        sns.histplot(data, bins=bins, kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    for j in range(i+1, rows*cols):
        axes[j].axis("off")
    st.pyplot(fig)
    st.divider()

# ---------- correlation heatmap ----------
st.subheader("Correlation Heatmap (Numeric Features)")
hm_df = dfw.select_dtypes(include=[np.number]).copy()
if drop_ids and id_cols:
    keep_cols = [c for c in hm_df.columns if c not in id_cols]
    hm_df = hm_df[keep_cols] if keep_cols else hm_df
if target_col and target_col in hm_df.columns:
    # keep target, but sort with others
    pass
if hm_df.shape[1] < 2:
    st.info("Not enough numeric columns for a correlation heatmap.", icon="ℹ️")
else:
    corr = hm_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(min(12, 1.2*corr.shape[1]), min(10, 0.6*corr.shape[0])))
    sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.3, ax=ax)
    ax.set_title("Correlation Heatmap (numeric)")
    st.pyplot(fig)
st.divider()

# ---------- Target vs Feature ----------
st.subheader("Target vs Feature")
if not target_col or target_col not in dfw.columns:
    st.info("Target column is not set. Go back to **Load Data** and save the target selection.", icon="ℹ️")
else:
    if feat_vs_target:
        if is_numeric_dtype(dfw[feat_vs_target]) and not is_bool_dtype(dfw[feat_vs_target]):
            # numeric feature
            if kind in ("Auto", "Box (num)"):
                fig, ax = plt.subplots(figsize=(7, 4.5))
                sns.boxplot(x=target_col, y=feat_vs_target, data=dfw, ax=ax)
                ax.set_title(f"{feat_vs_target} by {target_col}")
                st.pyplot(fig)
            else:
                st.caption("Selected plot type is for categorical; switched to box plot.")
        else:
            # categorical feature
            ct = (dfw.groupby([feat_vs_target, target_col]).size()
                  .reset_index(name="count"))
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.barplot(data=ct, y=feat_vs_target, x="count", hue=target_col, ax=ax)
            ax.set_title(f"{feat_vs_target} vs {target_col}")
            st.pyplot(fig)

st.divider()
st.subheader("Next step")
st.markdown("Go to **Train Model** from the sidebar or click the button below.")
if st.button("➡️ Proceed to Train Model"):
    try:
        st.switch_page("pages/5_Train_Model.py")
    except Exception:
        st.warning("Use the left sidebar → **5_Train_Model** to continue.")
