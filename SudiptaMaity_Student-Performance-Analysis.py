# =============================================================================
#  Student Performance Analysis & Academic Risk Prediction
#  Author  : Sudipta Maity
#  Dataset : student_info.csv  (1,000 student records)
#  Stack   : Streamlit · Pandas · Scikit-learn · Plotly
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f0f2f6; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .kpi-value { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.85rem; opacity: 0.85; margin: 4px 0 0; letter-spacing: 0.5px; }

    /* Section header */
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1e3a5f;
        border-left: 5px solid #2d6a9f; padding-left: 12px;
        margin: 24px 0 16px;
    }

    /* Risk badge */
    .risk-pass {
        background: #d1fae5; color: #065f46;
        border-radius: 8px; padding: 12px 20px;
        font-size: 1.2rem; font-weight: 700; text-align: center;
    }
    .risk-fail {
        background: #fee2e2; color: #991b1b;
        border-radius: 8px; padding: 12px 20px;
        font-size: 1.2rem; font-weight: 700; text-align: center;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #1e3a5f; }
    section[data-testid="stSidebar"] * { color: #e0ecff !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess(path: str = "student_info.csv"):
    df = pd.read_csv(path)

    # ── Numeric coercion ──────────────────────────────────────────────────────
    numeric_cols = ["age", "grade_level", "math_score", "reading_score",
                    "writing_score", "attendance_rate", "study_hours"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Fill missing values ───────────────────────────────────────────────────
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in ["gender", "parent_education", "internet_access",
                "lunch_type", "extra_activities", "final_result"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ── Derived features ──────────────────────────────────────────────────────
    df["avg_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1).round(2)
    df["pass_binary"] = (df["final_result"] == "Pass").astype(int)

    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a Random Forest classifier and return model + encoders + metrics."""
    feature_cols = [
        "age", "grade_level", "math_score", "reading_score", "writing_score",
        "attendance_rate", "study_hours", "avg_score",
        "gender", "parent_education", "internet_access",
        "lunch_type", "extra_activities"
    ]
    cat_cols = ["gender", "parent_education", "internet_access",
                "lunch_type", "extra_activities"]

    df_ml = df[feature_cols + ["final_result"]].copy()

    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    y = target_le.fit_transform(df_ml["final_result"])
    X = df_ml[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=target_le.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    return model, encoders, target_le, feature_cols, accuracy, report, cm, importances


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_and_preprocess("student_info.csv")
model, encoders, target_le, feature_cols, accuracy, report, cm, importances = train_model(df)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    
    st.markdown("## 🎓 Student Performance\nAnalysis & Risk Prediction")
    
    st.markdown("**Model:** Random Forest")
    

    # Global filter
    st.markdown("### 🔍 Global Filters")
    gender_filter = st.multiselect(
        "Gender",
        options=df["gender"].unique().tolist(),
        default=df["gender"].unique().tolist(),
    )
    grade_filter = st.multiselect(
        "Grade Level",
        options=sorted(df["grade_level"].unique().tolist()),
        default=sorted(df["grade_level"].unique().tolist()),
    )
    st.markdown("---")
    

# Apply filters
mask = (df["gender"].isin(gender_filter)) & (df["grade_level"].isin(grade_filter))
dff = df[mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#FF0000; margin-bottom:4px;'>"
    "🎓 Student Performance Analysis & Academic Risk Prediction"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#57606a; font-size:0.95rem;'>"
    "IBM SkillsBuild Capstone | Business Intelligence Dashboard | "
    "Author: <b>Sudipta Maity</b></p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview",
    "📈 Performance Drivers",
    "🤖 ML Model & Insights",
    "⚠️ Risk Predictor",
])


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── KPI Cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📌 Key Performance Indicators</div>',
                unsafe_allow_html=True)

    total_students  = len(dff)
    pass_rate       = (dff["pass_binary"].mean() * 100) if total_students > 0 else 0
    avg_attendance  = dff["attendance_rate"].mean() if total_students > 0 else 0
    avg_study_hours = dff["study_hours"].mean() if total_students > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{total_students:,}</p>'
            f'<p class="kpi-label">Total Students</p></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{pass_rate:.1f}%</p>'
            f'<p class="kpi-label">Overall Pass Rate</p></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{avg_attendance:.1f}%</p>'
            f'<p class="kpi-label">Avg Attendance Rate</p></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{avg_study_hours:.1f} h</p>'
            f'<p class="kpi-label">Avg Daily Study Hours</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pass / Fail Distribution ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Pass / Fail Distribution</div>',
                    unsafe_allow_html=True)
        result_counts = dff["final_result"].value_counts().reset_index()
        result_counts.columns = ["Result", "Count"]
        fig_pie = px.pie(
            result_counts, names="Result", values="Count",
            color="Result",
            color_discrete_map={"Pass": "#2d6a9f", "Fail": "#e05c5c"},
            hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                              pull=[0.03, 0.03])
        fig_pie.update_layout(showlegend=True, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Pass Rate by Grade Level</div>',
                    unsafe_allow_html=True)
        grade_pass = (
            dff.groupby("grade_level")["pass_binary"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"pass_binary": "Pass Rate (%)", "grade_level": "Grade Level"})
        )
        fig_grade = px.bar(
            grade_pass, x="Grade Level", y="Pass Rate (%)",
            color="Pass Rate (%)",
            color_continuous_scale="Blues",
            text="Pass Rate (%)",
        )
        fig_grade.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_grade.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_grade, use_container_width=True)

    # ── Pass Rate by Gender ────────────────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-header">Pass Rate by Gender</div>',
                    unsafe_allow_html=True)
        gender_pass = (
            dff.groupby("gender")["pass_binary"]
            .agg(["sum", "count"])
            .reset_index()
        )
        gender_pass["Pass Rate (%)"] = (gender_pass["sum"] / gender_pass["count"] * 100).round(1)
        fig_gen = px.bar(
            gender_pass, x="gender", y="Pass Rate (%)",
            color="gender",
            color_discrete_sequence=["#2d6a9f", "#e05c5c", "#f59e0b"],
            text="Pass Rate (%)",
        )
        fig_gen.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_gen.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_gen, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Score Distribution (Avg)</div>',
                    unsafe_allow_html=True)
        score_df = pd.melt(
            dff[["final_result", "math_score", "reading_score", "writing_score"]],
            id_vars="final_result",
            var_name="Subject",
            value_name="Score",
        )
        fig_box = px.box(
            score_df, x="Subject", y="Score", color="final_result",
            color_discrete_map={"Pass": "#2d6a9f", "Fail": "#e05c5c"},
        )
        fig_box.update_layout(margin=dict(t=20, b=20), legend_title="Result")
        st.plotly_chart(fig_box, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PERFORMANCE DRIVERS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">📚 Study Hours vs. Math Score</div>',
                unsafe_allow_html=True)
    fig_scatter = px.scatter(
        dff, x="study_hours", y="math_score",
        color="final_result",
        color_discrete_map={"Pass": "#2d6a9f", "Fail": "#e05c5c"},
        trendline="ols",
        labels={"study_hours": "Daily Study Hours", "math_score": "Math Score"},
        hover_data=["name", "attendance_rate", "avg_score"],
    )
    fig_scatter.update_layout(margin=dict(t=20, b=20), legend_title="Result")
    st.plotly_chart(fig_scatter, use_container_width=True)

    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<div class="section-header">Attendance Rate vs. Final Result</div>',
                    unsafe_allow_html=True)
        fig_att = px.violin(
            dff, x="final_result", y="attendance_rate",
            color="final_result",
            color_discrete_map={"Pass": "#2d6a9f", "Fail": "#e05c5c"},
            box=True, points="outliers",
            labels={"attendance_rate": "Attendance Rate (%)", "final_result": "Result"},
        )
        fig_att.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_att, use_container_width=True)

    with col_f:
        st.markdown('<div class="section-header">Parent Education vs. Pass Rate</div>',
                    unsafe_allow_html=True)
        edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
        edu_pass = (
            dff.groupby("parent_education")["pass_binary"]
            .mean()
            .mul(100)
            .reindex(edu_order, fill_value=0)
            .reset_index()
            .rename(columns={"pass_binary": "Pass Rate (%)", "parent_education": "Education Level"})
        )
        fig_edu = px.bar(
            edu_pass, x="Education Level", y="Pass Rate (%)",
            color="Pass Rate (%)",
            color_continuous_scale="teal",
            text="Pass Rate (%)",
            category_orders={"Education Level": edu_order},
        )
        fig_edu.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_edu.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_edu, use_container_width=True)

    # ── Internet Access & Lunch Type ───────────────────────────────────────────
    col_g, col_h = st.columns(2)

    with col_g:
        st.markdown('<div class="section-header">Internet Access Impact</div>',
                    unsafe_allow_html=True)
        net_pass = (
            dff.groupby(["internet_access", "final_result"])
            .size()
            .reset_index(name="Count")
        )
        fig_net = px.bar(
            net_pass, x="internet_access", y="Count", color="final_result",
            barmode="group",
            color_discrete_map={"Pass": "#2d6a9f", "Fail": "#e05c5c"},
            labels={"internet_access": "Internet Access", "Count": "Students"},
        )
        fig_net.update_layout(margin=dict(t=20, b=20), legend_title="Result")
        st.plotly_chart(fig_net, use_container_width=True)

    with col_h:
        st.markdown('<div class="section-header">Lunch Type vs. Avg Score</div>',
                    unsafe_allow_html=True)
        lunch_score = (
            dff.groupby("lunch_type")["avg_score"]
            .mean()
            .reset_index()
            .rename(columns={"avg_score": "Avg Score", "lunch_type": "Lunch Type"})
        )
        fig_lunch = px.bar(
            lunch_score, x="Lunch Type", y="Avg Score",
            color="Lunch Type",
            color_discrete_sequence=["#2d6a9f", "#f59e0b"],
            text="Avg Score",
        )
        fig_lunch.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_lunch.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig_lunch, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ML MODEL & INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">🤖 Random Forest Classifier — Performance</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    pass_report = report.get("Pass", {})
    fail_report = report.get("Fail", {})
    with m1:
        st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    with m2:
        st.metric("Pass Precision", f"{pass_report.get('precision', 0) * 100:.1f}%")
    with m3:
        st.metric("Fail Recall", f"{fail_report.get('recall', 0) * 100:.1f}%")
    with m4:
        st.metric("F1-Score (Pass)", f"{pass_report.get('f1-score', 0) * 100:.1f}%")

    col_i, col_j = st.columns(2)

    with col_i:
        st.markdown('<div class="section-header">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        labels_cm = list(target_le.classes_)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels_cm],
            y=[f"Actual {l}" for l in labels_cm],
            colorscale="Blues",
            text=cm.astype(str),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_cm.update_layout(margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_j:
        st.markdown('<div class="section-header">Top Feature Importances</div>',
                    unsafe_allow_html=True)
        imp_df = importances.head(10).reset_index()
        imp_df.columns = ["Feature", "Importance"]
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Classification Report Table ────────────────────────────────────────────
    st.markdown('<div class="section-header">Full Classification Report</div>',
                unsafe_allow_html=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Correlation Heatmap — Numeric Features</div>',
                unsafe_allow_html=True)
    num_cols = ["math_score", "reading_score", "writing_score",
                "attendance_rate", "study_hours", "avg_score", "pass_binary"]
    corr = dff[num_cols].corr().round(2)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        aspect="auto",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4 — LIVE RISK PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">⚠️ Live Student Academic Risk Predictor</div>',
                unsafe_allow_html=True)
    st.info(
        "Fill in a student's profile below to get a real-time **Pass / Fail** "
        "prediction powered by the trained Random Forest model.",
        icon="ℹ️",
    )

    with st.form("risk_form"):
        r1, r2, r3 = st.columns(3)

        with r1:
            p_age         = st.slider("Age", 13, 19, 16)
            p_grade       = st.selectbox("Grade Level", sorted(df["grade_level"].unique()))
            p_gender      = st.selectbox("Gender", sorted(df["gender"].unique()))
            p_internet    = st.selectbox("Internet Access", ["Yes", "No"])

        with r2:
            p_math        = st.slider("Math Score", 0, 100, 70)
            p_reading     = st.slider("Reading Score", 0, 100, 70)
            p_writing     = st.slider("Writing Score", 0, 100, 70)

        with r3:
            p_attendance  = st.slider("Attendance Rate (%)", 50.0, 100.0, 88.0, step=0.5)
            p_study       = st.slider("Study Hours / Day", 0.5, 6.0, 2.5, step=0.1)
            p_parent_edu  = st.selectbox("Parent Education",
                                         ["High School", "Bachelor's", "Master's", "PhD"])
            p_lunch       = st.selectbox("Lunch Type", ["Standard", "Free or reduced"])
            p_extra       = st.selectbox("Extra Activities", ["Yes", "No"])

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        avg_sc = round((p_math + p_reading + p_writing) / 3, 2)

        input_dict = {
            "age":               p_age,
            "grade_level":       p_grade,
            "math_score":        p_math,
            "reading_score":     p_reading,
            "writing_score":     p_writing,
            "attendance_rate":   p_attendance,
            "study_hours":       p_study,
            "avg_score":         avg_sc,
            "gender":            p_gender,
            "parent_education":  p_parent_edu,
            "internet_access":   p_internet,
            "lunch_type":        p_lunch,
            "extra_activities":  p_extra,
        }

        cat_cols_pred = ["gender", "parent_education", "internet_access",
                         "lunch_type", "extra_activities"]
        for col in cat_cols_pred:
            le = encoders[col]
            val = input_dict[col]
            if val in le.classes_:
                input_dict[col] = int(le.transform([val])[0])
            else:
                input_dict[col] = 0

        input_df   = pd.DataFrame([input_dict])[feature_cols]
        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        pred_label = target_le.inverse_transform([prediction])[0]

        classes    = list(target_le.classes_)
        pass_idx   = classes.index("Pass") if "Pass" in classes else 0
        fail_idx   = classes.index("Fail") if "Fail" in classes else 1
        pass_prob  = proba[pass_idx] * 100
        fail_prob  = proba[fail_idx] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])

        with res_col2:
            if pred_label == "Pass":
                st.markdown(
                    f'<div class="risk-pass">✅ Prediction: PASS &nbsp;|&nbsp; '
                    f'Confidence: {pass_prob:.1f}%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="risk-fail">🚨 Prediction: AT RISK (FAIL) &nbsp;|&nbsp; '
                    f'Fail Probability: {fail_prob:.1f}%</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        prob_fig = go.Figure(go.Bar(
            x=["Pass Probability", "Fail Probability"],
            y=[pass_prob, fail_prob],
            marker_color=["#2d6a9f", "#e05c5c"],
            text=[f"{pass_prob:.1f}%", f"{fail_prob:.1f}%"],
            textposition="outside",
        ))
        prob_fig.update_layout(
            title="Predicted Probability Breakdown",
            yaxis=dict(range=[0, 110], title="Probability (%)"),
            margin=dict(t=40, b=20),
            height=350,
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # ── Recommendation Panel ──────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Actionable Recommendations</div>',
                    unsafe_allow_html=True)
        recs = []
        if p_attendance < 85:
            recs.append("📅 **Attendance is low (<85%).** Engage counselling and attendance alerts.")
        if p_study < 2.0:
            recs.append("📖 **Study hours below 2 h/day.** Recommend structured study timetables.")
        if avg_sc < 60:
            recs.append("📝 **Average score below 60.** Schedule remedial/tutoring sessions.")
        if p_internet == "No":
            recs.append("🌐 **No internet access.** Provide offline resources or library access.")
        if not recs:
            recs.append("🌟 Student profile looks healthy — maintain current engagement levels.")
        for rec in recs:
            st.markdown(f"- {rec}")

    # ── Student Data Table ────────────────────────────────────────────────────
    with st.expander("📋 View Filtered Student Dataset"):
        st.dataframe(
            dff[[
                "student_id", "name", "gender", "grade_level",
                "math_score", "reading_score", "writing_score",
                "attendance_rate", "study_hours", "avg_score",
                "parent_education", "final_result"
            ]].reset_index(drop=True),
            use_container_width=True,
        )
