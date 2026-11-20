from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]

# Read the labeled dataset and its labels.
rides = pd.read_parquet(PROJECT_DIR / "GBG500.parquet")
labels = pd.read_csv(PROJECT_DIR / "GBG500_labels.csv")

# Compare ride IDs across the two files.
data_ids = set(rides["ride_id"].dropna())
label_ids = set(labels["RIDE_ID"].dropna())

print("Ride IDs in parquet:", len(data_ids))
print("Ride IDs in labels:", len(label_ids))
print("Parquet rides without labels:", len(data_ids - label_ids))
print("Labels without parquet rides:", len(label_ids - data_ids))

# Summarize each ride without mixing data from different rides.
rides["missing_sensor_row"] = rides[SENSORS].isna().any(axis=1)

summary = rides.groupby("ride_id").agg(
    samples=("time_index", "size"),
    missing_sensor_rows=("missing_sensor_row", "sum"),
    first_time=("time_index", "min"),
    last_time=("time_index", "max"),
)

summary["missing_percent"] = (
    100 * summary["missing_sensor_rows"] / summary["samples"]
)

summary = summary.merge(
    labels,
    left_index=True,
    right_on="RIDE_ID",
    how="left",
    validate="one_to_one",
)

print("\nSamples per ride:")
print(summary["samples"].describe().to_string())

print("\nMissing sensor data per ride (%):")
print(summary["missing_percent"].describe().to_string())

print("\nRides with any missing sensor data:")
print((summary["missing_sensor_rows"] > 0).sum())

print("\nRide counts by label:")
print(summary["LABEL"].value_counts(dropna=False).to_string())

print("\nMissing sensor data by label (%):")
print(summary.groupby("LABEL")["missing_percent"].describe().to_string())

print("\nDuplicate ride/time pairs:")
print(rides.duplicated(subset=["ride_id", "time_index"]).sum())