# 🌊 Flood Prediction System

A machine-learning Streamlit application that estimates flood risk from rainfall, river level, humidity, soil moisture, forecast rainfall, and drainage capacity.

## Features
- Flood / No Flood classification
- Estimated flood probability and risk band
- Interactive Streamlit dashboard
- Dataset charts and model test accuracy
- Reproducible sample dataset generation

## Technologies
Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib/Streamlit charts.

## Project structure
```text
FloodPredictionSystem/
├── app.py
├── data/
│   └── flood_data.csv
├── src/
│   ├── __init__.py
│   └── model.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

Optional: regenerate the included sample dataset with `python generate_data.py`.

## Important limitation
The included dataset is synthetic sample data created for an academic software demonstration. Model scores and predictions are not validated against live local flood observations. This application is not an official flood warning system and must not be used for emergency decisions. For real-world deployment, train and validate it with quality historical/local sensor data and integrate official weather and disaster-management alerts.
