import streamlit as st
import pandas as pd
from pathlib import Path
from src.model import train_model, FEATURES

st.set_page_config(page_title="Flood Prediction System", page_icon="🌊", layout="wide")

@st.cache_resource
def get_model():
    return train_model()

model, accuracy = get_model()
data = pd.read_csv(Path(__file__).parent / "data" / "flood_data.csv")

st.title("🌊 Flood Prediction System")
st.caption("Machine-learning demonstration for estimating flood risk from weather and environmental inputs.")
st.info("Demo notice: This project uses a sample synthetic dataset for academic demonstration. It is not an official warning service and must not be used for emergency decisions.")

c1, c2, c3 = st.columns(3)
c1.metric("Dataset records", f"{len(data):,}")
c2.metric("Model test accuracy", f"{accuracy*100:.1f}%")
c3.metric("Flood cases in sample", f"{int(data['flood'].sum()):,}")

st.divider()
st.subheader("🔎 Enter environmental conditions")
with st.form("prediction_form"):
    left, middle, right = st.columns(3)
    with left:
        rainfall = st.number_input("Current rainfall (mm)", 0.0, 300.0, 45.0, 1.0)
        river = st.number_input("River water level (m)", 0.0, 12.0, 3.5, 0.1)
    with middle:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 75.0, 1.0)
        soil = st.number_input("Soil moisture (%)", 0.0, 100.0, 60.0, 1.0)
    with right:
        forecast = st.number_input("Forecast rainfall (mm)", 0.0, 300.0, 55.0, 1.0)
        drainage = st.number_input("Drainage capacity (%)", 0.0, 100.0, 55.0, 1.0)
    submitted = st.form_submit_button("🌧️ Predict Flood Risk", use_container_width=True)

if submitted:
    values = pd.DataFrame([{
        "rainfall_mm": rainfall,
        "river_level_m": river,
        "humidity_pct": humidity,
        "soil_moisture_pct": soil,
        "forecast_rainfall_mm": forecast,
        "drainage_capacity_pct": drainage
    }], columns=FEATURES)
    probability = float(model.predict_proba(values)[0][1])
    prediction = int(model.predict(values)[0])
    st.subheader("📢 Prediction Result")
    a, b = st.columns(2)
    a.metric("Estimated flood probability", f"{probability*100:.1f}%")
    b.metric("Model classification", "Flood risk" if prediction else "No flood predicted")
    if probability >= 0.70:
        st.error("🔴 High estimated risk. Follow official local authority alerts and safety instructions.")
    elif probability >= 0.40:
        st.warning("🟠 Moderate estimated risk. Monitor official weather and water-level updates.")
    else:
        st.success("🟢 Lower estimated risk in this model output. Continue to follow official advisories.")
    st.progress(min(max(probability, 0.0), 1.0))

st.divider()
st.subheader("📊 Sample dataset overview")
col1, col2 = st.columns(2)
with col1:
    st.write("Flood / No Flood distribution")
    st.bar_chart(data["flood"].value_counts().rename(index={0:"No Flood", 1:"Flood"}))
with col2:
    st.write("Rainfall vs river level")
    st.scatter_chart(data, x="rainfall_mm", y="river_level_m", color="flood")
st.caption("Model metrics are calculated from a held-out test split of the included sample dataset.")
