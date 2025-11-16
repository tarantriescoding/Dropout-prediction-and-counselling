# app.py
# UmeedRise – Simple, Trustworthy Student Dropout Prediction & Support
# Designed for schools, teachers, students, and communities with clear language and guided steps.
# Author: Copilot

import os
import io
import json
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
import shap
import plotly.graph_objs as go
import plotly.express as px
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------
# Page config & accessible UI
# -----------------------------
st.set_page_config(
    page_title="UmeedRise – Simple Student Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light gradient & accessible visual style
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E8ECFF 0%, #C7C3F5 100%);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 18px;
        box-shadow: 0 8px 28px rgba(31, 38, 135, 0.20);
        backdrop-filter: blur(7px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 18px;
        margin-bottom: 18px;
    }
    h1, h2, h3 {
        color: #20204A;
        font-weight: 700;
        letter-spacing: 0.2px;
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
    }
    p, span, label, div, li {
        color: #1F2756;
        font-size: 16px;
        line-height: 1.6;
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(6px);
        border-right: 1px solid rgba(255,255,255,0.3);
    }
    .stButton>button, .stDownloadButton>button {
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 16px;
        border: none;
    }
    .stButton>button {
        background-color: #5B6EF0;
        color: white;
    }
    .stDownloadButton>button {
        background-color: #4FB37A;
        color: white;
    }
    .alert-banner {
        border-left: 6px solid #D8345F;
        background: rgba(216, 52, 95, 0.14);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .medium-banner {
        border-left: 6px solid #F2A007;
        background: rgba(242, 160, 7, 0.15);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .success-banner {
        border-left: 6px solid #4FB37A;
        background: rgba(79, 179, 122, 0.15);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .badge-low { background: #DDF7E7; color: #0F6A44; }
    .badge-medium { background: #FFF2CC; color: #6C4A00; }
    .badge-high { background: #FFE3E8; color: #9B1028; }
    .stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }
    .helper {
        font-size: 14px;
        color: #3A3A70;
        background: rgba(255,255,255,0.55);
        border: 1px dashed rgba(0,0,0,0.15);
        padding: 10px;
        border-radius: 10px;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Language support (English/Hindi)
# -----------------------------
LANG = st.sidebar.selectbox("Language / भाषा", ["English", "हिंदी"])

def T(en: str, hi: str) -> str:
    return hi if LANG == "हिंदी" else en

# -----------------------------
# Sidebar navigation & simple mode
# -----------------------------
st.sidebar.title("🎓 UmeedRise")
st.sidebar.markdown(T("Simple student support for schools & communities.", "स्कूल और समुदायों के लिए सरल छात्र सहायता।"))

mode = st.sidebar.radio(T("Mode", "मोड"), options=[T("Simple", "सरल"), T("Advanced", "उन्नत")], index=0)

page = st.sidebar.radio(
    T("Navigate", "नेविगेट करें"),
    options=[
        T("Home", "होम"),
        T("Upload & Train", "अपलोड करें और ट्रेन करें"),
        T("Dashboard", "डैशबोर्ड"),
        T("Student Search", "छात्र खोज"),
        T("Explainability (SHAP)", "व्याख्येयता (SHAP)"),
        T("Counseling & Mentoring", "परामर्श और साथियों का मार्गदर्शन"),
    ],
    index=0,
)

# Risk thresholds & hyperparameters (simple defaults vs advanced controls)
st.sidebar.markdown("---")
st.sidebar.subheader(T("Settings", "सेटिंग्स"))
if mode == T("Simple", "सरल"):
    low_th = 0.40
    high_th = 0.70
    alert_high = 0.80
    params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.08, "scale_pos_weight": 1.0}
    st.sidebar.markdown(T("Using safe defaults for accuracy and stability.", "शुद्धता और स्थिरता के लिए सुरक्षित डिफ़ॉल्ट का उपयोग किया जा रहा है।"))
else:
    low_th = st.sidebar.slider(T("Low risk threshold", "कम जोखिम सीमा"), 0.1, 0.6, 0.4, 0.05)
    high_th = st.sidebar.slider(T("High risk threshold", "उच्च जोखिम सीमा"), 0.5, 0.95, 0.7, 0.05)
    alert_high = st.sidebar.slider(T("Alert high-risk cutoff", "अलर्ट उच्च जोखिम कटऑफ"), 0.7, 0.95, 0.80, 0.01)
    st.sidebar.markdown(T("XGBoost Hyperparameters", "XGBoost हाइपरपैरामीटर"))
    n_estimators = st.sidebar.number_input("n_estimators", 50, 1000, 300, 50)
    max_depth = st.sidebar.number_input("max_depth", 2, 12, 5, 1)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.08, 0.01)
    scale_pos_weight = st.sidebar.slider(T("scale_pos_weight (class imbalance)", "scale_pos_weight (कक्षा असंतुलन)"), 0.5, 10.0, 1.0, 0.5)
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "scale_pos_weight": float(scale_pos_weight),
    }

# -----------------------------
# State
# -----------------------------
for key in ["df", "detected", "train_result", "pred_df", "risk_labels", "alerts_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

MODEL_PATH = "umeedrise_model.joblib"
PREPROCESSOR_PATH = "umeedrise_preprocessor.joblib"
METADATA_PATH = "umeedrise_metadata.json"

# -----------------------------
# Helpers
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _normalized(col: str) -> str:
    return col.strip().lower().replace(" ", "").replace("_", "")

def load_data(uploaded_file) -> pd.DataFrame:
    """
    Robust CSV loader with safe fallbacks.
    """
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1", on_bad_lines="skip")
    except Exception:
        data = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(data), on_bad_lines="skip")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def detect_columns(df: pd.DataFrame) -> Dict[str, any]:
    """
    Auto-detect target, ID, numeric, categorical, attendance, and marks columns.
    """
    cols = df.columns.tolist()
    normalized_map = {c: _normalized(c) for c in cols}

    target_candidates = ["dropout", "label", "target", "isdropout", "risk", "y"]
    detected_target = None
    for c, norm in normalized_map.items():
        if any(norm == _normalized(tc) for tc in target_candidates):
            detected_target = c
            break
    if detected_target is None:
        for c in cols:
            unique = df[c].dropna().unique()
            if len(unique) <= 3:
                vals = set([str(v).strip().lower() for v in unique])
                binary_like = vals.issubset({"0", "1", "true", "false", "yes", "no"})
                if binary_like:
                    detected_target = c
                    break

    id_candidates = ["studentid","id","roll","rollno","rollnumber","email","admissionno","admno","regno","registrationno","name","fullname"]
    detected_id = None
    for c, norm in normalized_map.items():
        if any(norm == _normalized(ic) for ic in id_candidates):
            detected_id = c
            break
    if detected_id is None:
        for c in cols:
            if df[c].dtype == "object" and df[c].nunique() > max(10, int(0.2 * len(df))):
                detected_id = c
                break

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]

    if detected_target in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != detected_target]
    if detected_target in categorical_cols:
        categorical_cols = [c for c in categorical_cols if c != detected_target]

    attendance_cols = [c for c in cols if "attendance" in _normalized(c)]
    marks_cols = [c for c in cols if any(k in _normalized(c) for k in ["marks","score","grade","percentage"])]

    return {
        "target": detected_target,
        "id_col": detected_id,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "attendance_cols": attendance_cols,
        "marks_cols": marks_cols,
    }

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """
    Numeric: Median Imputer + StandardScaler
    Categorical: Most-frequent Imputer + OneHotEncoder(sparse_output=False)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor

def _make_binary_target(y: pd.Series) -> np.ndarray:
    """
    Convert target to 0/1 consistently from various formats.
    """
    y_clean = y.copy()
    if y_clean.dtype == "object":
        y_lower = y_clean.astype(str).str.strip().str.lower()
        y_bin = y_lower.map(
            lambda v: 1 if v in ["1","true","yes","y","dropout","high","risk"] else 0
        )
    else:
        y_bin = y_clean.map(lambda v: 1 if str(v).strip().lower() in ["1","true"] else 0)
    y_bin = y_bin.fillna(0).astype(int).values
    return y_bin

def _auto_scale_pos_weight(y: np.ndarray) -> float:
    """
    Auto compute scale_pos_weight for class imbalance: (negatives / positives).
    """
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return 1.0
    return max(1.0, float(neg) / float(pos))

def train_xgboost(
    df: pd.DataFrame,
    target_col: str,
    id_col: Optional[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Dict[str, any],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Train XGBClassifier with robust preprocessing and artifact saving.
    """
    X = df[numeric_cols + categorical_cols].copy()
    y = _make_binary_target(df[target_col])

    # Auto-adjust scale_pos_weight if simple mode
    if mode == T("Simple", "सरल"):
        params["scale_pos_weight"] = _auto_scale_pos_weight(y)

    stratify = y if len(np.unique(y)) > 1 else None
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify, random_state=random_state
    )

    model = XGBClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        max_depth=int(params.get("max_depth", 5)),
        learning_rate=float(params.get("learning_rate", 0.08)),
        scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="auto",
        use_label_encoder=False
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    model.fit(X_train_proc, y_train)

    # Feature names
    try:
        num_features = numeric_cols
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        ohe_features = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names = num_features + ohe_features
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train_proc.shape[1])]

    # Save artifacts
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        meta = {
            "target_col": target_col,
            "id_col": id_col,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names": feature_names,
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(meta, f)
    except Exception:
        pass

    return {
        "preprocessor": preprocessor,
        "model": model,
        "X_train": X_train,
        "X_train_proc": X_train_proc,
        "X_test": X_test,
        "X_test_proc": X_test_proc,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
    }

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan
    return metrics

def predict_student(
    model: XGBClassifier,
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X_proc = preprocessor.transform(X)
    prob = model.predict_proba(X_proc)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return pd.DataFrame({"probability": prob, "prediction": pred})

def risk_label(prob: float, thresholds: Tuple[float, float] = (0.4, 0.7)) -> str:
    low, high = thresholds
    if prob < low:
        return "Low"
    elif prob < high:
        return "Medium"
    else:
        return "High"

def explain_shap(
    model: XGBClassifier,
    X_proc: np.ndarray,
    feature_names: List[str],
    index: Optional[int] = None,
) -> Dict[str, any]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs}).sort_values("importance", ascending=False)
    sample_idx = index if index is not None else 0
    sample_shap = shap_values[sample_idx]
    base_value = explainer.expected_value
    return {"shap_values": shap_values, "importance_df": importance_df, "sample_shap": sample_shap, "base_value": base_value}

def generate_counseling(risk: str) -> List[str]:
    if risk == "Low":
        return [T("Continue consistent study habits", "लगातार अध्ययन आदतें जारी रखें"),
                T("Maintain attendance", "उपस्थिति बनाए रखें")]
    elif risk == "Medium":
        return [T("Teacher mentoring", "शिक्षक का मार्गदर्शन"),
                T("Assignment discipline", "असाइनमेंट अनुशासन"),
                T("Light peer mentoring", "हल्का साथियों का मार्गदर्शन")]
    else:
        return [T("Immediate counselor meeting", "काउंसलर से तुरंत मिलें"),
                T("Personalized academic plan", "व्यक्तिगत शैक्षणिक योजना"),
                T("Emotional support", "भावनात्मक सहयोग"),
                T("Weekly follow-up", "साप्ताहिक फॉलो-अप"),
                T("Mandatory peer mentoring assignment", "अनिवार्य साथियों का मार्गदर्शन असाइनमेंट")]

def _attendance_summary(df: pd.DataFrame, attendance_cols: List[str]) -> Optional[str]:
    if not attendance_cols:
        return None
    try:
        vals = df[attendance_cols].applymap(safe_float).mean(axis=1)
        return T(f"Avg attendance ~ {np.nanmean(vals):.1f}", f"औसत उपस्थिति ~ {np.nanmean(vals):.1f}")
    except Exception:
        return None

def _marks_summary(df: pd.DataFrame, marks_cols: List[str]) -> Optional[str]:
    if not marks_cols:
        return None
    try:
        vals = df[marks_cols].applymap(safe_float).mean(axis=1)
        return T(f"Avg marks ~ {np.nanmean(vals):.1f}", f"औसत अंक ~ {np.nanmean(vals):.1f}")
    except Exception:
        return None

def make_confusion_matrix_plot(cm: np.ndarray) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred 0", "Pred 1"],
        y=["True 0", "True 1"],
        colorscale="Blues",
        showscale=True
    ))
    fig.update_layout(title=T("Confusion Matrix","कन्फ्यूज़न मैट्रिक्स"),
                      xaxis_title=T("Predicted","अनुमानित"),
                      yaxis_title=T("Actual","वास्तविक"),
                      margin=dict(l=0, r=0, t=30, b=0), height=350)
    return fig

def make_risk_distribution_plot(probs: np.ndarray) -> go.Figure:
    fig = px.histogram(x=probs, nbins=30, color_discrete_sequence=["#5B6EF0"])
    fig.update_layout(title=T("Risk probability distribution","जोखिम संभावना वितरण"),
                      xaxis_title=T("Dropout probability","ड्रॉपआउट संभावना"),
                      yaxis_title=T("Count","गिनती"),
                      margin=dict(l=0, r=0, t=30, b=0), height=350)
    return fig

def make_risk_pie(labels: List[str]) -> go.Figure:
    fig = px.pie(names=labels, title=T("Risk level composition","जोखिम स्तर संरचना"), color=labels,
                 color_discrete_map={"Low":"#4FB37A","Medium":"#F2A007","High":"#D8345F"})
    fig.update_layout(height=350)
    return fig

def make_importance_bar(df_imp: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = df_imp.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=data["importance"], y=data["feature"], orientation="h", marker_color="#5B6EF0"))
    fig.update_layout(title=T(f"Top {top_n} features by SHAP importance", f"SHAP महत्त्व के अनुसार शीर्ष {top_n} फीचर्स"),
                      xaxis_title=T("Mean |SHAP|","औसत |SHAP|"),
                      yaxis_title=T("Feature","फीचर"),
                      margin=dict(l=0, r=0, t=30, b=0), height=500)
    return fig

def build_alerts(pred_df: pd.DataFrame, thresholds: Tuple[float, float], alert_high: float, df: pd.DataFrame, attendance_cols: List[str]) -> pd.DataFrame:
    attend_flag = None
    if attendance_cols:
        try:
            att_val = df[attendance_cols].applymap(safe_float).mean(axis=1)
            attend_flag = att_val < np.nanpercentile(att_val, 20)
        except Exception:
            attend_flag = None

    alerts = []
    for i, r in pred_df.iterrows():
        if r["probability"] >= alert_high:
            alerts.append((T(f"High Risk ≥ {alert_high:.2f}", f"उच्च जोखिम ≥ {alert_high:.2f}"), i, "High"))
        if thresholds[0] <= r["probability"] < thresholds[1]:
            alerts.append((T("Repeated Medium Risk (pattern)", "बार-बार मध्यम जोखिम (पैटर्न)"), i, "Medium"))
        if attend_flag is not None and attend_flag.iloc[i]:
            alerts.append((T("Low Attendance", "कम उपस्थिति"), i, "Medium"))
    return pd.DataFrame(alerts, columns=["Alert", "Index", "Severity"])

def style_risk_cell(label: str) -> str:
    if label == "Low":
        text = T("Low","कम")
        return f'<span class="badge badge-low">{text}</span>'
    elif label == "Medium":
        text = T("Medium","मध्यम")
        return f'<span class="badge badge-medium">{text}</span>'
    else:
        text = T("High","उच्च")
        return f'<span class="badge badge-high">{text}</span>'

# -----------------------------
# Home
# -----------------------------
if page == T("Home","होम"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.title("UmeedRise")
    st.subheader(T("Simple, clear student dropout prediction & support", "सरल और स्पष्ट छात्र ड्रॉपआउट भविष्यवाणी और सहायता"))
    st.write(T(
        "Follow three steps: 1) Upload data 2) Train model 3) See dashboard & student support.",
        "तीन चरणों का पालन करें: 1) डेटा अपलोड करें 2) मॉडल ट्रेन करें 3) डैशबोर्ड और छात्र सहायता देखें।"
    ))
    st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🔄 " + T("Auto-detection","ऑटो-डिटेक्शन"))
            st.write(T("Target, ID, numeric, categorical columns detected safely.", "टार्गेट, आईडी, न्यूमेरिक और कैटेगोरिकल कॉलम सुरक्षित रूप से पहचानें।"))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 " + T("Explainable AI","व्याख्येय एआई"))
            st.write(T("See which features affect risk (SHAP).", "देखें कौन से फीचर जोखिम को प्रभावित करते हैं (SHAP)।"))
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🤝 " + T("Alerts & Mentoring","अलर्ट और मार्गदर्शन"))
            st.write(T("Get alerts and easy counseling plans; auto peer mentoring.", "अलर्ट और सरल परामर्श योजनाएँ; ऑटो साथियों का मार्गदर्शन।"))
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card helper">', unsafe_allow_html=True)
    st.write(T("Tip: Start in Simple mode for one-click training.", "टिप: एक-क्लिक ट्रेनिंग के लिए सरल मोड से शुरू करें।"))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Upload & Train
# -----------------------------
elif page == T("Upload & Train","अपलोड करें और ट्रेन करें"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁 " + T("Upload dataset","डेटासेट अपलोड करें"))
    uploaded_file = st.file_uploader(T("Upload CSV file","CSV फ़ाइल अपलोड करें"), type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧭 " + T("Auto-detection results","ऑटो-डिटेक्शन परिणाम"))
        detected = detect_columns(df)
        st.session_state.detected = detected

        colA, colB, colC = st.columns(3)
        with colA:
            st.write(f"**{T('Label:','लेबल:')}** " + T("Target column","टार्गेट कॉलम"))
            st.info(detected["target"] if detected["target"] else T("Not found – please select below","नहीं मिला – नीचे चयन करें"))
        with colB:
            st.write(f"**{T('Label:','लेबल:')}** " + T("ID column","आईडी कॉलम"))
            st.info(detected["id_col"] if detected["id_col"] else T("Not found – optional","नहीं मिला – वैकल्पिक"))
        with colC:
            st.write(f"**{T('Label:','लेबल:')}** " + T("Feature counts","फीचर गणना"))
            st.info(T(f"Numeric: {len(detected['numeric_cols'])}, Categorical: {len(detected['categorical_cols'])}",
                      f"न्यूमेरिक: {len(detected['numeric_cols'])}, कैटेगोरिकल: {len(detected['categorical_cols'])}"))
        st.markdown('</div>', unsafe_allow_html=True)

        # Manual overrides (kept simple)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🛠️ " + T("Confirm or adjust columns","कॉलम की पुष्टि/समायोजन करें"))
        target_col = st.selectbox(T("Select target column (0/1, Yes/No preferred)","टार्गेट कॉलम चुनें (0/1, हाँ/नहीं बेहतर)"),
                                  options=df.columns.tolist(),
                                  index=(df.columns.tolist().index(detected["target"]) if detected["target"] in df.columns else 0))
        id_col = st.selectbox(T("Select ID column (optional)","आईडी कॉलम चुनें (वैकल्पिक)"),
                              options=["None"] + df.columns.tolist(),
                              index=(df.columns.tolist().index(detected["id_col"]) + 1 if detected["id_col"] in df.columns else 0))

        all_feature_cols = [c for c in df.columns if c != target_col]
        default_num = [c for c in detected["numeric_cols"] if c in all_feature_cols]
        default_cat = [c for c in detected["categorical_cols"] if c in all_feature_cols and c != target_col]

        if mode == T("Simple","सरल"):
            # Hide complexity: use detected automatically
            numeric_cols = default_num
            categorical_cols = default_cat
            st.write(T("Using auto-selected features.","ऑटो-चयनित फीचर्स का उपयोग किया जा रहा है।"))
        else:
            numeric_cols = st.multiselect(T("Numeric feature columns","न्यूमेरिक फीचर कॉलम"), options=all_feature_cols, default=default_num)
            categorical_cols = st.multiselect(T("Categorical feature columns","कैटेगोरिकल फीचर कॉलम"),
                                              options=[c for c in all_feature_cols if c not in numeric_cols], default=default_cat)
        st.markdown('</div>', unsafe_allow_html=True)

        # Preview & stats (simple)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 👀 " + T("Preview & stats","प्रीव्यू और आँकड़े"))
        st.dataframe(df.head(15), use_container_width=True)
        st.write(T(f"Rows: {len(df)}, Columns: {len(df.columns)}", f"पंक्तियाँ: {len(df)}, कॉलम: {len(df.columns)}"))
        st.write("**" + T("Missing values per column","प्रति कॉलम मिसिंग वैल्यू") + "**")
        st.dataframe(df.isna().sum().to_frame(T("missing","मिसिंग")).sort_values(T("missing","मिसिंग"), ascending=False), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Train button
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 " + T("Train model","मॉडल ट्रेन करें"))
        label_train = T("One-click Train","एक-क्लिक ट्रेन")
        if st.button(label_train):
            if len(numeric_cols) + len(categorical_cols) == 0:
                st.error(T("Please select at least one feature column.","कृपया कम से कम एक फीचर कॉलम चुनें।"))
            else:
                try:
                    id_selected = None if id_col == "None" else id_col
                    train_res = train_xgboost(df, target_col, id_selected, numeric_cols, categorical_cols, params)
                    st.session_state.train_result = train_res

                    # Evaluate
                    y_test = train_res["y_test"]
                    X_test_proc = train_res["X_test_proc"]
                    model = train_res["model"]
                    y_prob = model.predict_proba(X_test_proc)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    metrics = evaluate_model(y_test, y_pred, y_prob)

                    st.success(T("Model trained successfully.","मॉडल सफलतापूर्वक ट्रेन हुआ।"))
                    st.write(f"**{T('Accuracy','सटीकता')}:** {metrics['accuracy']:.3f}")
                    st.write(f"**{T('Precision','प्रिसीजन')}:** {metrics['precision']:.3f}")
                    st.write(f"**{T('Recall','रिकॉल')}:** {metrics['recall']:.3f}")
                    st.write(f"**{T('F1-score','F1-स्कोर')}:** {metrics['f1']:.3f}")
                    st.write(f"**{T('ROC-AUC','ROC-AUC')}:** {metrics['roc_auc']:.3f}")

                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    st.plotly_chart(make_confusion_matrix_plot(cm), use_container_width=True)

                    # Full predictions
                    feature_cols = numeric_cols + categorical_cols
                    full_pred = predict_student(model, train_res["preprocessor"], df, feature_cols)
                    risk_labels = [risk_label(p, thresholds=(low_th, high_th)) for p in full_pred["probability"]]
                    st.session_state.pred_df = full_pred
                    st.session_state.risk_labels = risk_labels

                    # Alerts
                    alerts_df = build_alerts(full_pred, (low_th, high_th), alert_high, df, detected["attendance_cols"])
                    st.session_state.alerts_df = alerts_df

                    merged = df.copy()
                    merged["dropout_probability"] = full_pred["probability"]
                    merged["risk_level"] = risk_labels

                    st.download_button(T("Download predictions CSV","प्रेडिक्शन्स CSV डाउनलोड करें"),
                                       data=merged.to_csv(index=False).encode("utf-8"),
                                       file_name="umeedrise_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(T(f"Training failed: {e}", f"ट्रेनिंग विफल: {e}"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info(T("Upload a CSV to begin.","शुरू करने के लिए CSV अपलोड करें।"))

# -----------------------------
# Dashboard
# -----------------------------
elif page == T("Dashboard","डैशबोर्ड"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 " + T("Risk analytics dashboard","जोखिम विश्लेषण डैशबोर्ड"))

    if st.session_state.df is None or st.session_state.pred_df is None:
        st.info(T("Please upload and train a model first.","कृपया पहले अपलोड करें और मॉडल ट्रेन करें।"))
    else:
        df = st.session_state.df
        pred_df = st.session_state.pred_df
        risk_labels = st.session_state.risk_labels
        detected = st.session_state.detected
        feature_names = (st.session_state.train_result or {}).get("feature_names", [])
        alerts_df = st.session_state.alerts_df

        if alerts_df is not None and len(alerts_df) > 0:
            st.markdown('<div class="alert-banner">', unsafe_allow_html=True)
            st.markdown(f"**{T('Alerts','अलर्ट')}:** {len(alerts_df)}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-banner">', unsafe_allow_html=True)
            st.markdown("**" + T("Status: No critical alerts","स्थिति: कोई गंभीर अलर्ट नहीं") + "**")
            st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### " + T("Risk probability distribution","जोखिम संभावना वितरण"))
            st.plotly_chart(make_risk_distribution_plot(pred_df["probability"].values), use_container_width=True)
        with c2:
            st.markdown("#### " + T("Composition of risk levels","जोखिम स्तरों की संरचना"))
            st.plotly_chart(make_risk_pie(risk_labels), use_container_width=True)

        if st.session_state.train_result is not None and len(feature_names) > 0:
            st.markdown("#### " + T("Feature importance (SHAP)","फीचर महत्त्व (SHAP)"))
            try:
                train_res = st.session_state.train_result
                imp_df = explain_shap(train_res["model"], train_res["X_train_proc"], feature_names)["importance_df"]
                st.plotly_chart(make_importance_bar(imp_df, top_n=20), use_container_width=True)
            except Exception:
                st.info(T("Feature importance not available.","फीचर महत्त्व उपलब्ध नहीं।"))

        st.markdown("#### " + T("Top high-risk students","शीर्ष उच्च-जोखिम छात्र"))
        merged = df.copy()
        merged["dropout_probability"] = pred_df["probability"].values
        merged["risk_level"] = risk_labels
        merged_sorted = merged.sort_values("dropout_probability", ascending=False)
        top_n = st.slider(T("How many to show?","कितने दिखाएँ?"), 5, 100, 20, 5)
        st.dataframe(merged_sorted.head(top_n), use_container_width=True)

        st.markdown("#### " + T("Alerts panel","अलर्ट पैनल"))
        if alerts_df is not None and len(alerts_df) > 0:
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.write(T("No alerts.","कोई अलर्ट नहीं।"))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Student Search
# -----------------------------
elif page == T("Student Search","छात्र खोज"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔎 " + T("Student search & instant explanations","छात्र खोज और तात्क्षणिक व्याख्या"))

    if st.session_state.df is None or st.session_state.pred_df is None or st.session_state.train_result is None:
        st.info(T("Please upload and train a model first.","कृपया पहले अपलोड करें और मॉडल ट्रेन करें।"))
    else:
        df = st.session_state.df
        pred_df = st.session_state.pred_df
        risk_labels = st.session_state.risk_labels
        detected = st.session_state.detected
        id_col = detected["id_col"] if detected else None
        feature_cols = (st.session_state.train_result["X_train"].columns.tolist()
                        if st.session_state.train_result is not None else [])
        feature_names = st.session_state.train_result["feature_names"]

        query = st.text_input(T("Search by Student ID, Name, Roll No, or Email","छात्र आईडी, नाम, रोल नंबर या ईमेल से खोजें"))
        if query:
            qnorm = query.strip().lower()
            candidates = []
            id_candidates = [id_col] if id_col else []
            id_candidates += [c for c in df.columns if any(k in _normalized(c) for k in ["id","roll","email","name","adm","reg"])]
            id_candidates = list(dict.fromkeys([c for c in id_candidates if c in df.columns]))

            for c in id_candidates:
                try:
                    mask = df[c].astype(str).str.lower().str.contains(qnorm, na=False)
                    idxs = df.index[mask].tolist()
                    for ix in idxs:
                        candidates.append(ix)
                except Exception:
                    continue
            candidates = list(dict.fromkeys(candidates))

            if len(candidates) == 0:
                st.warning(T("No matching students found.","कोई मिलते-जुलते छात्र नहीं मिले।"))
            else:
                st.success(T(f"Found {len(candidates)} record(s).", f"{len(candidates)} रिकॉर्ड मिले।"))
                sel_ix = st.selectbox(T("Select a record index","रिकॉर्ड इंडेक्स चुनें"), options=candidates, index=0)

                prob = pred_df.loc[sel_ix, "probability"]
                label = risk_label(prob, thresholds=(low_th, high_th))
                badge_html = style_risk_cell(label)
                st.markdown(f"**{T('Risk level','जोखिम स्तर')}:** {badge_html} — **{T('Probability','संभावना')}:** {prob:.3f}", unsafe_allow_html=True)

                att_summary = _attendance_summary(df, detected["attendance_cols"])
                mk_summary = _marks_summary(df, detected["marks_cols"])
                if att_summary or mk_summary:
                    st.markdown("#### " + T("Summary","सार"))
                    if att_summary: st.write(f"**{T('Label:','लेबल:')}** {att_summary}")
                    if mk_summary: st.write(f"**{T('Label:','लेबल:')}** {mk_summary}")

                st.markdown("#### " + T("SHAP per-student explanation","SHAP प्रति-छात्र व्याख्या"))
                try:
                    train_res = st.session_state.train_result
                    X_row = df.loc[[sel_ix], feature_cols]
                    X_row_proc = train_res["preprocessor"].transform(X_row)
                    shap_res = explain_shap(train_res["model"], X_row_proc, feature_names, index=0)

                    # Waterfall (matplotlib)
                    st.write("**" + T("Contribution to dropout probability","ड्रॉपआउट संभावना में योगदान") + "**")
                    plt.figure(figsize=(8,6))
                    shap.plots._waterfall.waterfall_legacy(
                        shap_res["base_value"],
                        shap_res["sample_shap"],
                        feature_names=feature_names,
                        max_display=12,
                        show=False
                    )
                    st.pyplot(plt.gcf(), use_container_width=True)
                    plt.close()

                    # Top contributors bar (plotly)
                    order = np.argsort(-np.abs(shap_res["sample_shap"]))
                    top_df = pd.DataFrame({"feature": np.array(feature_names)[order][:12],
                                           "shap_value": shap_res["sample_shap"][order][:12]})
                    fig_bar = go.Figure(go.Bar(
                        x=top_df["shap_value"],
                        y=top_df["feature"],
                        orientation="h",
                        marker_color=["#D8345F" if v > 0 else "#4FB37A" for v in top_df["shap_value"]]
                    ))
                    fig_bar.update_layout(
                        title=T("Top feature contributions (positive = increase risk)","शीर्ष फीचर योगदान (सकारात्मक = जोखिम बढ़ता है)"),
                        xaxis_title="SHAP",
                        yaxis_title=T("Feature","फीचर"),
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    st.markdown("#### " + T("Counseling suggestions","परामर्श सुझाव"))
                    for s in generate_counseling(label):
                        st.write(f"- {s}")

                except Exception as e:
                    st.error(T(f"SHAP explanation unavailable: {e}", f"SHAP व्याख्या उपलब्ध नहीं: {e}"))

                student_row = df.loc[[sel_ix]].copy()
                student_row["dropout_probability"] = prob
                student_row["risk_level"] = label
                st.download_button(T("Download this student's report (CSV)","इस छात्र की रिपोर्ट डाउनलोड करें (CSV)"),
                                   data=student_row.to_csv(index=False).encode("utf-8"),
                                   file_name=f"student_{sel_ix}_report.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Explainability (SHAP)
# -----------------------------
elif page == T("Explainability (SHAP)","व्याख्येयता (SHAP)"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 " + T("Global explainability with SHAP","SHAP के साथ वैश्विक व्याख्येयता"))

    if st.session_state.train_result is None:
        st.info(T("Train a model first.","पहले मॉडल ट्रेन करें।"))
    else:
        train_res = st.session_state.train_result
        feature_names = train_res["feature_names"]
        X_train_proc = train_res["X_train_proc"]
        model = train_res["model"]

        try:
            shap_res = explain_shap(model, X_train_proc, feature_names)
            imp_df = shap_res["importance_df"]

            st.write("**" + T("SHAP summary (beeswarm)","SHAP सारांश (बीस्वॉर्म)") + "**")
            plt.figure(figsize=(9,6))
            shap.summary_plot(shap_res["shap_values"], X_train_proc, feature_names=feature_names, show=False)
            st.pyplot(plt.gcf(), use_container_width=True)
            plt.close()

            st.write("**" + T("SHAP global bar chart","SHAP वैश्विक बार चार्ट") + "**")
            st.plotly_chart(make_importance_bar(imp_df, top_n=25), use_container_width=True)
        except Exception as e:
            st.error(T(f"Unable to render SHAP plots: {e}", f"SHAP प्लॉट रेंडर नहीं कर सके: {e}"))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Counseling & Mentoring
# -----------------------------
elif page == T("Counseling & Mentoring","परामर्श और साथियों का मार्गदर्शन"):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🤝 " + T("Counseling & peer mentoring","परामर्श और साथियों का मार्गदर्शन"))

    if st.session_state.df is None or st.session_state.pred_df is None:
        st.info(T("Train a model to generate counseling and mentoring.","परामर्श और मार्गदर्शन के लिए मॉडल ट्रेन करें।"))
    else:
        df = st.session_state.df
        pred_df = st.session_state.pred_df
        risk_labels = st.session_state.risk_labels
        detected = st.session_state.detected
        id_col = detected["id_col"]

        merged = df.copy()
        merged["dropout_probability"] = pred_df["probability"].values
        merged["risk_level"] = risk_labels

        high_df = merged[merged["risk_level"] == "High"].sort_values("dropout_probability", ascending=False)
        med_df = merged[merged["risk_level"] == "Medium"].sort_values("dropout_probability", ascending=False)

        st.markdown("#### " + T("High-risk – mandatory peer mentoring","उच्च जोखिम – अनिवार्य साथियों का मार्गदर्शन"))
        if len(high_df) == 0:
            st.write(T("No high-risk students at the moment.","इस समय कोई उच्च-जोखिम छात्र नहीं।"))
        else:
            show_cols = [id_col] + [c for c in merged.columns if c != id_col] if id_col in merged.columns else merged.columns
            st.dataframe(high_df[show_cols].head(50), use_container_width=True)

        st.markdown("#### " + T("Medium-risk – suggested group mentoring","मध्यम जोखिम – समूह मार्गदर्शन"))
        if len(med_df) == 0:
            st.write(T("No medium-risk students at the moment.","इस समय कोई मध्यम-जोखिम छात्र नहीं।"))
        else:
            show_cols = [id_col] + [c for c in merged.columns if c != id_col] if id_col in merged.columns else merged.columns
            st.dataframe(med_df[show_cols].head(50), use_container_width=True)

        st.markdown("#### " + T("Mentor assignment logic","मार्गदर्शक असाइनमेंट लॉजिक"))
        st.write(T(
            "We suggest mentors who have good attendance and marks, and pair them with students from similar classes.",
            "हम उन साथियों को मार्गदर्शक सुझाते हैं जिनकी उपस्थिति और अंक अच्छे हैं, और उन्हें समान कक्षा के छात्रों से जोड़ते हैं।"
        ))

        att_cols = detected["attendance_cols"]
        mk_cols = detected["marks_cols"]
        potential_mentors = merged.copy()
        try:
            att_val = potential_mentors[att_cols].applymap(safe_float).mean(axis=1) if att_cols else 0
            mk_val = potential_mentors[mk_cols].applymap(safe_float).mean(axis=1) if mk_cols else 0
            potential_mentors["mentor_score"] = 0.5 * (att_val if hasattr(att_val, "fillna") else att_val) + \
                                                0.5 * (mk_val if hasattr(mk_val, "fillna") else mk_val)
        except Exception:
            potential_mentors["mentor_score"] = 0.5

        mentor_df = potential_mentors[potential_mentors["risk_level"] == "Low"].sort_values("mentor_score", ascending=False).head(50)
        id_like_cols = [c for c in merged.columns if any(k in _normalized(c) for k in ["id","name","email","roll"])]
        cols_show = list(dict.fromkeys(id_like_cols + ["risk_level","dropout_probability","mentor_score"]))
        st.markdown("#### " + T("Potential peer mentors","संभावित मार्गदर्शक"))
        st.dataframe(mentor_df[cols_show] if len(cols_show) > 0 else mentor_df.head(20), use_container_width=True)

        st.markdown("#### " + T("Auto-match mentors to high-risk students (demo)","उच्च-जोखिम छात्रों से मार्गदर्शक का ऑटो-मैच (डेमो)"))
        try:
            assign_count = min(len(high_df), len(mentor_df), 20)
            matches = []
            for i in range(assign_count):
                mentee_row = high_df.iloc[i]
                mentor_row = mentor_df.iloc[i]
                mentee_id = mentee_row[id_col] if id_col in high_df.columns else f"Index {mentee_row.name}"
                mentor_id = mentor_row[id_col] if id_col in mentor_df.columns else f"Index {mentor_row.name}"
                matches.append({T("Mentee","मेंटे"): mentee_id,
                                T("Mentor","मार्गदर्शक"): mentor_id,
                                T("Mentor score","मार्गदर्शक स्कोर"): mentor_row.get("mentor_score", np.nan)})
            if len(matches) > 0:
                st.dataframe(pd.DataFrame(matches), use_container_width=True)
            else:
                st.write(T("Not enough mentors to match.","मैच करने के लिए पर्याप्त मार्गदर्शक नहीं।"))
        except Exception:
            st.write(T("Mentor matching unavailable for this dataset; please ensure ID columns are present.",
                       "इस डेटासेट के लिए मैचिंग उपलब्ध नहीं; कृपया आईडी कॉलम सुनिश्चित करें।"))
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📝 " + T("Notes","नोट्स"))
st.write(T(
    "This app uses a robust pipeline and XGBoost for accuracy. SHAP explains which features impact risk. Alerts trigger for high risk, repeated medium risk, and low attendance.",
    "यह ऐप सटीकता के लिए मजबूत पाइपलाइन और XGBoost का उपयोग करता है। SHAP बताता है कि कौन से फीचर जोखिम को प्रभावित करते हैं। अलर्ट उच्च जोखिम, बार-बार मध्यम जोखिम और कम उपस्थिति पर ट्रिगर होते हैं।"
))
st.markdown('</div>', unsafe_allow_html=True)
