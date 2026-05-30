# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from app import predict_student_result

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.main-title {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    color: #1E3A8A;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #555;
    margin-bottom: 40px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        '<div class="main-title">🎓 Student Performance Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Based Academic Analysis Dashboard</div>',
        unsafe_allow_html=True
    )

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=200
    )

    st.write("### 📘 Project Features")
    st.write("""
    - Predict PASS or FAIL
    - Analyze academic performance
    - AI-powered recommendations
    - Interactive charts and dashboard
    """)

    if st.button("🚀 Start Prediction", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()

# =========================================================
# INPUT PAGE
# =========================================================

elif st.session_state.page == "input":

    st.title("📥 Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider(
            "📌 Attendance (%)",
            0,
            100,
            75
        )

        study_hours = st.slider(
            "📚 Study Hours Per Day",
            0,
            12,
            4
        )

        internal_marks = st.slider(
            "📝 Internal Marks",
            0,
            100,
            65
        )

    with col2:

        assignment_completion = st.slider(
            "📂 Assignment Completion (%)",
            0,
            100,
            70
        )

        previous_score = st.slider(
            "📊 Previous Score",
            0,
            100,
            68
        )

    if st.button("🎯 Predict Result", use_container_width=True):

        st.session_state.attendance = attendance
        st.session_state.study_hours = study_hours
        st.session_state.internal_marks = internal_marks
        st.session_state.assignment_completion = assignment_completion
        st.session_state.previous_score = previous_score

        st.session_state.page = "result"
        st.rerun()

# =========================================================
# RESULT PAGE
# =========================================================

elif st.session_state.page == "result":

    attendance = st.session_state.attendance
    study_hours = st.session_state.study_hours
    internal_marks = st.session_state.internal_marks
    assignment_completion = st.session_state.assignment_completion
    previous_score = st.session_state.previous_score

    st.title("🎯 Prediction Result")

    summary_df = pd.DataFrame({
        "Feature": [
            "Attendance",
            "Study Hours",
            "Internal Marks",
            "Assignment Completion",
            "Previous Score"
        ],
        "Value": [
            attendance,
            study_hours,
            internal_marks,
            assignment_completion,
            previous_score
        ]
    })

    st.dataframe(summary_df, use_container_width=True)

    fig = px.bar(
        summary_df,
        x="Feature",
        y="Value",
        text="Value",
        title="Student Performance Indicators"
    )

    st.plotly_chart(fig, use_container_width=True)

    result, confidence = predict_student_result(
        attendance,
        study_hours,
        internal_marks,
        assignment_completion,
        previous_score
    )

    performance_score = (
        attendance +
        internal_marks +
        assignment_completion +
        previous_score
    ) / 4

    st.progress(int(performance_score))

    st.success(
        f"Overall Academic Performance Score: {performance_score:.2f}%"
    )

    if result.lower() == "pass":

        st.success("✅ Student is likely to PASS")
        st.balloons()

    else:

        st.error("❌ Student is likely to FAIL")

    st.info(f"Prediction Confidence: {confidence:.2f}%")

    # =====================================================
    # SUGGESTIONS
    # =====================================================

    st.subheader("💡 Improvement Suggestions")

    suggestions = []

    if attendance < 75:
        suggestions.append("Increase attendance percentage.")

    if study_hours < 3:
        suggestions.append("Spend more time studying daily.")

    if internal_marks < 50:
        suggestions.append("Improve internal exam preparation.")

    if assignment_completion < 60:
        suggestions.append("Complete assignments on time.")

    if previous_score < 50:
        suggestions.append("Focus on weak subjects.")

    if suggestions:

        for suggestion in suggestions:
            st.warning(suggestion)

    else:
        st.success("Excellent academic performance! 🎉")

    # =====================================================
    # BACK BUTTON streamlit run gui.py
    # =====================================================

    if st.button("🔙 Predict Again", use_container_width=True):
        st.session_state.page = "input"
        st.rerun()