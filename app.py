import streamlit as st

st.set_page_config(
    page_title="Student Placement Predictor",
    layout="wide"
)

import joblib
import pandas as pd
import numpy as np

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_models():
    cls_model = joblib.load("best_classification.pkl")
    reg_model = joblib.load("best_regression.pkl")
    return cls_model, reg_model

cls_model, reg_model = load_models()

# =========================
# UI CONFIG
# =========================


st.title("🎓 Student Placement & Salary Prediction")
st.markdown("Predict placement status dan estimasi gaji berdasarkan data mahasiswa")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input Features")

def user_input():
    age = st.sidebar.slider("Age", 18, 35, 22)
    gpa = st.sidebar.slider("GPA", 0.0, 4.0, 3.0)
    internship = st.sidebar.selectbox("Internship", ["Yes", "No"])
    projects = st.sidebar.slider("Projects", 0, 10, 2)
    certifications = st.sidebar.slider("Certifications", 0, 10, 1)

    data = {
        "age": age,
        "gpa": gpa,
        "internship": internship,
        "projects": projects,
        "certifications": certifications
    }

    return pd.DataFrame([data])

input_df = user_input()

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Data")
st.write(input_df)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    # Predict
    pred_cls = cls_model.predict(input_df)[0]
    pred_reg = reg_model.predict(input_df)[0]

    # Mapping label (jika encode sebelumnya)
    placement = "Placed" if pred_cls == 1 else "Not Placed"

    # =========================
    # OUTPUT
    # =========================
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Placement Status", placement)

    with col2:
        st.metric("Estimated Salary (LPA)", f"{pred_reg:.2f}")

    # =========================
    # VISUALIZATION
    # =========================
    st.subheader("📈 Simple Visualization")

    chart_data = pd.DataFrame({
        "Metric": ["GPA", "Projects", "Certifications"],
        "Value": [input_df["gpa"][0], input_df["projects"][0], input_df["certifications"][0]]
    })

    st.bar_chart(chart_data.set_index("Metric"))