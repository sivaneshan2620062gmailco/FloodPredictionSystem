"""Regenerate the sample synthetic flood dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2500
rain = rng.gamma(2.2, 18, n).clip(0, 250)
river = rng.normal(3.0, 1.4, n).clip(0.2, 9.5)
humidity = rng.normal(72, 13, n).clip(25, 100)
soil = rng.normal(55, 20, n).clip(5, 100)
forecast = (rain * rng.uniform(.15, 1.25, n) + rng.normal(0, 12, n)).clip(0, 300)
drainage = rng.normal(55, 18, n).clip(5, 100)
score = .27*rain/250 + .26*river/9.5 + .12*humidity/100 + .13*soil/100 + .17*forecast/300 + .05*(1-drainage/100)
flood = ((score + rng.normal(0, .055, n)) > .47).astype(int)
df = pd.DataFrame({
    "rainfall_mm": rain.round(1), "river_level_m": river.round(2),
    "humidity_pct": humidity.round(1), "soil_moisture_pct": soil.round(1),
    "forecast_rainfall_mm": forecast.round(1),
    "drainage_capacity_pct": drainage.round(1), "flood": flood
})
out = Path(__file__).parent / "data" / "flood_data.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
