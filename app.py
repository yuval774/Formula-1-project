import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openai

st.set_page_config(page_title="Formula 1 Analysis", layout="wide")

# Set your OpenAI API key via Streamlit secrets
api_key = st.secrets.get("openai_api_key")
if api_key:
    openai.api_key = api_key
else:
    openai.api_key = ""

def ask_gpt(prompt):
    if not openai.api_key:
        return "❌ Error: OpenAI API key not found."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

@st.cache_data

def load_data():
    results = pd.read_csv("https://drive.google.com/uc?id=1Y8oD8HQLnXzzQIaJkJBwvUN7Ds9E4OCx&export=download")
    qualifying = pd.read_csv("https://drive.google.com/uc?id=1mcakLSYRJvoq-Am7Cpo8Be-qFD6bYl7M&export=download")
    return results, qualifying

results, qualifying = load_data()

def lap_time_to_seconds_safe(x):
    try:
        minutes, seconds = x.split(":")
        return float(minutes) * 60 + float(seconds)
    except:
        return None

qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds_safe)
q1_cleaned = qualifying[qualifying["q1_seconds"].notna()]
filtered_results = results[(results["grid"] >= 1) & (results["grid"] <= 20)]
results["grid_group"] = results["grid"].apply(lambda x: "Top 5 grid" if x <= 5 else "Grid 6–20")
results["position_change"] = results["grid"] - results["positionOrder"]
lap_data = results.copy()
lap_data = lap_data[lap_data["rank"].notna() & lap_data["positionOrder"].notna()]
lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)
top_drivers = results["driverId"].value_counts().head(6).index
top_driver_data = results[results["driverId"].isin(top_drivers)]

# Sidebar fun facts
st.sidebar.markdown("""
### 🧠 Did You Know?
- A pole position increases win chance by 40%
- Fastest lap doesn’t guarantee a podium
- Most races are decided in the first 5 laps
""")

# Overview and interactive graph selector
tabs = st.tabs(["🏁 Overview", "📊 Visualizations"])

with tabs[
