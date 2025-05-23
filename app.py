import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openai

st.set_page_config(page_title="Formula 1 Analysis", layout="wide")

# Set your OpenAI API key via Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

def ask_gpt(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load data from Google Drive links
@st.cache_data
def load_data():
    results = pd.read_csv("https://drive.google.com/uc?id=1Y8oD8HQLnXzzQIaJkJBwvUN7Ds9E4OCx&export=download")
    qualifying = pd.read_csv("https://drive.google.com/uc?id=1mcakLSYRJvoq-Am7Cpo8Be-qFD6bYl7M&export=download")
    return results, qualifying

results, qualifying = load_data()

# Fixing lap time conversion with safe parsing
def lap_time_to_seconds_safe(x):
    try:
        parts = x.split(":")
        return float(parts[0])*60 + float(parts[1])
    except:
        return np.nan

qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds_safe)
q1_cleaned = qualifying[qualifying["q1_seconds"].notna()]

# Tabs
overview_tab, graphs_tab = st.tabs(["🏁 Project Overview", "📊 Data Visualizations"])

# Sidebar additions
st.sidebar.markdown("""
### 🧠 Did You Know?
- A pole position increases win chance by 40%
- Fastest lap doesn’t guarantee a podium
- Most races are decided in the first 5 laps
""")

# --- Tab 1: Overview ---
with overview_tab:
    st.title("Formula 1 Performance Analysis")
    st.markdown("#### It’s lights out and away we goooo!")
    st.markdown("""
    **Submitted by:** Yuval Vin  
    **Track:** Business Administration with Digital Innovation  

    This project explores the impact of qualifying positions, fastest laps, and race strategy on final results in Formula 1.  
    We examine how starting grid positions relate to finishing positions, analyze driver consistency, and investigate race volatility.
    """)

    st.markdown("---")
    st.subheader("📊 Summary Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Drivers", len(results["driverId"].unique()))
    col2.metric("Total Races", len(results["raceId"].unique()))
    col3.metric("Fastest Q1 Time (s)", round(q1_cleaned["q1_seconds"].min(), 2))

# --- Tab 2: Graphs ---
with graphs_tab:
    st.subheader("Q1 Lap Time Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(q1_cleaned["q1_seconds"], bins=30, kde=True, ax=ax, color="purple")
    ax.set_xlabel("Q1 Time (seconds)")
    ax.set_ylabel("Number of Drivers")
    st.pyplot(fig)
    st.markdown("Most drivers clock lap times between 78–100 seconds. The slight right skew indicates a few slower performances.")
