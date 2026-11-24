"""Streamlit web app to test the model through the API.

Run: API_URL=http://localhost:8000 streamlit run webapp/app.py
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ride classifier", layout="wide")
st.title("Ride classifier — GBG500")

try:
    info = requests.get(f"{API_URL}/model", timeout=5).json()
except requests.RequestException as e:
    st.error(f"API unreachable at {API_URL}: {e}")
    st.stop()

st.caption(f"Model version {info['version']} · runtime {info['runtime']} · features: {', '.join(info['features'])}")

st.write("Upload the IMU readings of **one ride** (one row per reading). "
         "Sample files: `python scripts/export_sample_rides.py` writes test rides to `data/samples/`.")
uploaded = st.file_uploader("CSV with columns " + ", ".join(info["features"]), type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    missing = [c for c in info["features"] if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    col1, col2 = st.columns(2)
    col1.subheader(f"Sensor signals ({len(df)} readings)")
    col1.line_chart(df[info["features"]])
    if "true_label" in df.columns:
        col1.caption(f"True label (sample file): {df['true_label'].iloc[0]}")
    if st.button("Predict", type="primary"):
        points = df[info["features"]].astype(float).fillna(0.0).values.tolist()
        r = requests.post(f"{API_URL}/predict", json={"points": points}, timeout=30)
        if r.ok:
            res = r.json()
            col2.metric("Prediction", res["label"], f"{res['confidence']:.1%} confidence")
            col2.bar_chart(pd.Series(res["probabilities"], name="probability"))
        else:
            st.error(f"{r.status_code}: {r.text}")
