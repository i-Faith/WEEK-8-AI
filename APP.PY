import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# Load the Trained Model 
with open("VITAL_model.pkl", "rb") as f:
    model = pickle.load(f)

#  Page Config & Styling 
st.set_page_config(page_title="AI Health Monitor", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background color: white;
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'TIMES NEW ROMAN', sans-serif;
        text-align: center;
    }
    .block-container {
        backdrop-filter: blur(4px);
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: black;

    .stTitle {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333;
    .stHeader {
        font-size: 1.5rem; 
        font-weight: bold;
        color: #555;
    .stSubheader {
        font-size: 1.2rem;
        color: #666;
    .stMarkdown {
        font-size: 1rem;   
        color: #444;
    . stButton {
        background-color: #4CAF50;
        color: white;
        font-size: 1rem;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;

      
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("AI-POWERED HEALTH MONITORING SYSTEM")
st.markdown("Predict health risk based on real-time and historical vital signs.")

# Sidebar: User Input 
st.sidebar.header("Input Your Vital Signs")

def user_input():
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 40, 180, 72)
    bp_sys = st.sidebar.slider("Systolic BP (mmHg)", 80, 200, 120)
    bp_dia = st.sidebar.slider("Diastolic BP (mmHg)", 40, 130, 80)
    resp_rate = st.sidebar.slider("Respiratory Rate", 10, 40, 16)
    temp = st.sidebar.slider("Body Temperature (°C)", 34.0, 42.0, 36.8)
    spo2 = st.sidebar.slider("SpO2 (%)", 70, 100, 98)

    data = {
        "Heart Rate": heart_rate,
        "Systolic Blood Pressure": bp_sys,
        "Diastolic Blood Pressure": bp_dia,
        "Respiratory Rate": resp_rate,
        "Body Temperature": temp,
        "Oxygen Saturation": spo2
    }

    return pd.DataFrame([data])

input_df = user_input()

# =Display Input 
st.subheader("Entered Vital Signs")
st.dataframe(input_df)

# Predict Risk Level 
if st.button("Predict Risk Level"):
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"**Risk Level:** {prediction}")

    # Alert Messages
    if prediction == "High Risk":
        st.error("🚨 High Risk Detected!")
        st.markdown("**Recommendation:** Seek immediate medical attention.")
    elif prediction == "Moderate Risk":
        st.warning("⚠️ Moderate Risk Detected.")
        st.markdown("**Recommendation:** Monitor closely and consult your doctor.")
    else:
        st.success("✅ Normal Health Status")
        st.markdown("**Keep up the healthy lifestyle!**")
else:
    prediction = None  # Avoid error in later use

# Real-Time Simulated Metrics
st.header("Real-Time Health Metrics")

placeholder = st.empty()
for _ in range(10):
    with placeholder.container():
        st.metric("Heart Rate", f"{np.random.randint(60, 100)} bpm")
        st.metric("SpO2", f"{np.random.randint(90, 100)} %")
        st.metric("Temperature", f"{round(np.random.uniform(36.0, 37.5), 1)} °C")
    time.sleep(1)

# Upload and Display Historical Data
st.header("Historical Data Viewer")
uploaded_file = st.file_uploader("Upload your health data (CSV)", type=["csv"])

if uploaded_file is not None:
    history = pd.read_csv(uploaded_file)

    st.write("## Raw Data")
    st.dataframe(history)

    if "HeartRate" in history.columns:
        st.write("### Heart Rate Over Time")
        st.line_chart(history["HeartRate"])

    if "Temperature" in history.columns and "SpO2" in history.columns:
        st.write("### Temperature & SpO2 Trends")
        st.area_chart(history[["Temperature", "SpO2"]])
    else:
        st.warning("Ensure 'Temperature' and 'SpO2' columns exist in your CSV.")
