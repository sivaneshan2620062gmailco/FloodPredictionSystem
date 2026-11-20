import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

FEATURES = [
    "rainfall_mm", "river_level_m", "humidity_pct",
    "soil_moisture_pct", "forecast_rainfall_mm",
    "drainage_capacity_pct"
]

def train_model(csv_path=None):
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "flood_data.csv"
    data = pd.read_csv(csv_path)
    X = data[FEATURES]
    y = data["flood"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy
