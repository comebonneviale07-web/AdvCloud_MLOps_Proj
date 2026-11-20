from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]

rides = pd.read_parquet(PROJECT_DIR / "GBG500.parquet")
labels = pd.read_csv(PROJECT_DIR / "GBG500_labels.csv")

# Average sensor readings that share a ride ID and timestamp.
rides = (
    rides.groupby(["ride_id", "time_index"], as_index=False)[SENSORS]
    .mean()
)

# Calculate one set of sensor statistics per ride.
# These statistics ignore individual missing readings.
features = rides.groupby("ride_id")[SENSORS].agg(
    ["mean", "std", "min", "max", "median"]
)
features.columns = [
    f"{sensor}_{stat}" for sensor, stat in features.columns
]
features = features.reset_index()

# Attach the class label to each ride.
features = features.merge(
    labels,
    left_on="ride_id",
    right_on="RIDE_ID",
    how="inner",
    validate="one_to_one",
).drop(columns="RIDE_ID")

print("Feature table shape:", features.shape)
print("\nRides per class:")
print(features["LABEL"].value_counts().to_string())
print("\nMissing feature values:")
print(features.isna().sum().to_string())

features.to_csv(PROJECT_DIR / "ride_features.csv", index=False)
print("\nSaved ride_features.csv")