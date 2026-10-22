from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------- Paths and settings --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_training_data.csv"
MODEL_PATH = BASE_DIR / "model" / "performance_model.joblib"

st.set_page_config(
    page_title="EduTrack AI | Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Professional theme --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0B1220 0%, #101B2D 100%);
        color: #E5EDF7;
    }
    [data-testid="stSidebar"] {
        background: #0F1929;
        border-right: 1px solid #263449;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #67E8F9;
    }
    h1 { color: #F8FAFC !important; font-weight: 750 !important; }
    h2, h3 { color: #67E8F9 !important; }
    p, label, .stMarkdown { color: #CBD5E1; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #18283D, #132033);
        border: 1px solid #2A3D55;
        padding: 18px;
        border-radius: 15px;
    }
    [data-testid="stMetricLabel"] { color: #A9BCD2 !important; }
    [data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 750; }
    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        background: #0891B2;
        color: white;
        border: 1px solid #22D3EE;
        border-radius: 10px;
        font-weight: 650;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #0E7490; color: white; border-color: #67E8F9;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #2A3D55; border-radius: 12px;
    }
    hr { border-color: #263449; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Data helpers --------------------
DEFAULT_COLUMNS = [
    "Student_ID", "Student", "Attendance", "Study_Hours",
    "Previous_Marks", "Marks"
]

def make_demo_data():
    return pd.DataFrame([
        {"Student_ID": 1, "Student": "Aarav Sharma", "Attendance": 92, "Study_Hours": 3.0, "Previous_Marks": 78, "Marks": 82},
        {"Student_ID": 2, "Student": "Ananya Patil", "Attendance": 85, "Study_Hours": 2.5, "Previous_Marks": 72, "Marks": 76},
        {"Student_ID": 3, "Student": "Rohan Deshmukh", "Attendance": 68, "Study_Hours": 1.0, "Previous_Marks": 54, "Marks": 58},
        {"Student_ID": 4, "Student": "Isha Kulkarni", "Attendance": 96, "Study_Hours": 4.0, "Previous_Marks": 88, "Marks": 91},
        {"Student_ID": 5, "Student": "Kabir Joshi", "Attendance": 74, "Study_Hours": 1.5, "Previous_Marks": 61, "Marks": 64},
        {"Student_ID": 6, "Student": "Meera Shinde", "Attendance": 89, "Study_Hours": 3.5, "Previous_Marks": 81, "Marks": 85},
        {"Student_ID": 7, "Student": "Vivaan Pawar", "Attendance": 61, "Study_Hours": 0.8, "Previous_Marks": 48, "Marks": 51},
        {"Student_ID": 8, "Student": "Saanvi More", "Attendance": 91, "Study_Hours": 2.8, "Previous_Marks": 84, "Marks": 86},
        {"Student_ID": 9, "Student": "Aditya Jadhav", "Attendance": 79, "Study_Hours": 2.0, "Previous_Marks": 66, "Marks": 69},
        {"Student_ID": 10, "Student": "Diya Bansode", "Attendance": 87, "Study_Hours": 3.2, "Previous_Marks": 75, "Marks": 80},
    ])
def load_data():
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = make_demo_data()
    else:
        df = make_demo_data()

    # Support slightly different column names from an existing dataset.
    aliases = {
        "student": "Student",
        "student_name": "Student",
        "name": "Student",
        "attendance_percentage": "Attendance",
        "attendance_pct": "Attendance",
        "study_hours": "Study_Hours",
        "previous_marks": "Previous_Marks",
        "previous_score": "Previous_Marks",
        "marks": "Marks",
        "score": "Marks",
    }
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in aliases and col != aliases[key]:
            rename[col] = aliases[key]
    if rename:
        df = df.rename(columns=rename)

    if "Student" not in df.columns:
        df["Student"] = [f"Student {i+1}" for i in range(len(df))]
    for col in ["Attendance", "Study_Hours", "Previous_Marks", "Marks"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "Student_ID" not in df.columns:
        df.insert(0, "Student_ID", range(1, len(df) + 1))
    df["Student_ID"] = pd.to_numeric(df["Student_ID"], errors="coerce")
    if df["Student_ID"].isna().any():
        df["Student_ID"] = range(1, len(df) + 1)
    return df

def save_data(df):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

def performance_label(attendance, study_hours, previous_marks):
    # Fallback only if the trained model file is not available.
    score = (float(attendance) * 0.25) + (float(study_hours) * 10) + (float(previous_marks) * 0.45)
    if score >= 70:
        return "High"
    if score >= 48:
        return "Medium"
    return "Needs Support"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None

def predict_performance(attendance, study_hours, previous_marks):
    attendance = float(attendance)
    study_hours = float(study_hours)
    previous_marks = float(previous_marks)

    model = load_model()

    if model is not None:
        try:
            features = pd.DataFrame([{
                "Attendance": attendance,
                "Study_Hours": study_hours,
                "Previous_Marks": previous_marks,
            }])

            prediction = model.predict(features)[0]
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                confidence = round(float(max(probabilities)) * 100, 2)

            return str(prediction), confidence, True

        except Exception as error:
            st.warning(f"ML prediction failed: {error}")

    result = performance_label(attendance, study_hours, previous_marks)
    return result, None, False


# -------------------- Load data --------------------
df = load_data()

# -------------------- Sidebar navigation --------------------
st.sidebar.markdown("# 🎓 EduTrack AI")
st.sidebar.caption("Student Performance & Analytics")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Student Explorer", "Manage Students", "Performance Prediction"],
)
st.sidebar.divider()
st.sidebar.caption("Portfolio demo • Local data storage")

# -------------------- Overview --------------------
if page == "Overview":
    st.title("EduTrack AI")
    st.markdown("### Student Performance Overview")
    st.caption("Explore student records, understand performance patterns, and review prediction estimates.")

    total_students = len(df)
    avg_attendance = float(df["Attendance"].mean()) if total_students else 0
    avg_marks = float(df["Marks"].mean()) if total_students else 0
    avg_study = float(df["Study_Hours"].mean()) if total_students else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", f"{total_students:,}")
    c2.metric("Avg. Attendance", f"{avg_attendance:.1f}%")
    c3.metric("Average Marks", f"{avg_marks:.1f}")
    c4.metric("Daily Study Hours", f"{avg_study:.1f}")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Marks Distribution")
        if total_students:
            fig = px.histogram(df, x="Marks", nbins=12, title="Student marks", color_discrete_sequence=["#22D3EE"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E5EDF7")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No student records yet.")
    with right:
        st.subheader("Attendance vs Marks")
        if total_students:
            fig = px.scatter(
                df, x="Attendance", y="Marks", hover_name="Student",
                title="Attendance and marks relationship",
                color_discrete_sequence=["#34D399"],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E5EDF7")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No student records yet.")

    st.subheader("Recent Student Records")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption("Demo note: use real, authorized data only. Predictions are educational estimates, not official academic decisions.")
# -------------------- Student Explorer --------------------
elif page == "Student Explorer":
    st.title("Student Explorer")
    st.markdown("### Search and review student records")
    search_text = st.text_input("Search by student name or ID", placeholder="Type a name or student ID...")
    filtered = df.copy()
    if search_text.strip():
        text = search_text.strip().lower()
        filtered = filtered[
            filtered["Student"].astype(str).str.lower().str.contains(text, na=False)
            | filtered["Student_ID"].astype(str).str.contains(text, na=False)
        ]

    st.write(f"**{len(filtered)}** record(s) found")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download displayed records as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="edutrack_student_records.csv",
        mime="text/csv",
    )

# -------------------- Manage Students --------------------
elif page == "Manage Students":
    st.title("Manage Student Records")
    st.markdown("### Edit existing records or add a new student")
    st.info("Edit values in the table, then click **Save Changes**. Saved records are written to data/student_training_data.csv.")

    st.subheader("1. Edit existing records")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="student_editor",
        column_config={
            "Student_ID": st.column_config.NumberColumn("Student ID", min_value=1, step=1, disabled=True),
            "Student": st.column_config.TextColumn("Student Name", required=True),
            "Attendance": st.column_config.NumberColumn("Attendance (%)", min_value=0, max_value=100, step=1, format="%.1f"),
            "Study_Hours": st.column_config.NumberColumn("Study Hours / Day", min_value=0, max_value=24, step=0.5, format="%.1f"),
            "Previous_Marks": st.column_config.NumberColumn("Previous Marks", min_value=0, max_value=100, step=1, format="%.1f"),
            "Marks": st.column_config.NumberColumn("Current Marks", min_value=0, max_value=100, step=1, format="%.1f"),
        },
    )
    if st.button("💾 Save Changes", type="primary", key="save_student_changes"):
        edited_df["Student"] = edited_df["Student"].astype(str).str.strip()
        if (edited_df["Student"] == "").any():
            st.error("Student name cannot be empty. Please correct it and save again.")
        elif edited_df["Student_ID"].duplicated().any():
            st.error("Student IDs must be unique.")
        else:
            save_data(edited_df)
            st.success("Student records saved successfully!")
            st.rerun()

    st.divider()
    st.subheader("2. Add a new student")
    with st.form("add_student_form", clear_on_submit=True):
        new_name = st.text_input("Student name")
        col1, col2 = st.columns(2)
        with col1:
            new_attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
            new_study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
        with col2:
            new_previous_marks = st.number_input("Previous marks", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
            new_marks = st.number_input("Current marks", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        submitted = st.form_submit_button("➕ Add Student")
        if submitted:
            if not new_name.strip():
                st.error("Please enter the student's name.")
            else:
                next_id = int(pd.to_numeric(df["Student_ID"], errors="coerce").max() or 0) + 1
                new_row = {
                    "Student_ID": next_id,
                    "Student": new_name.strip(),
                    "Attendance": new_attendance,
                    "Study_Hours": new_study_hours,
                    "Previous_Marks": new_previous_marks,
                    "Marks": new_marks,
                }
                updated = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(updated)
                st.success(f"{new_name.strip()} added successfully!")
                st.rerun()

    st.caption("Tip: Back up your CSV before making many changes. Records are saved locally on this computer.")
# -------------------- Performance Prediction --------------------
elif page == "Performance Prediction":
    st.title("Performance Prediction")
    st.markdown("### Estimate a student's performance category")
    st.write("Enter student details below and click the prediction button.")

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=80)
        study_hours = st.slider("Daily study hours", min_value=0.0, max_value=12.0, value=2.0, step=0.5)
    with col2:
        previous_marks = st.slider("Previous marks (%)", min_value=0, max_value=100, value=65)

    if st.button("✨ Predict Performance", type="primary"):
        result, confidence, used_model = predict_performance(attendance, study_hours, previous_marks)
        st.subheader("Prediction Result")
        st.metric("Estimated Performance", result)
        if confidence is not None:
            st.progress(int(max(0, min(100, confidence))))
            st.write(f"Model confidence: **{confidence:.1f}%**")
        if used_model:
            st.success("Prediction generated using the saved machine-learning model.")
        else:
            st.warning("Trained model not found or could not be loaded. Showing a simple demo estimate instead. Run `python train_model.py` from the project folder to train the model.")
        st.caption("This is a portfolio demonstration using synthetic/demo data. Do not use it for real academic decisions.")

st.divider()
st.caption("EduTrack AI • Student Analytics Portfolio Project")
