import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Formula 1 Analysis", layout="wide")

# Load data from Google Drive links
@st.cache_data
def load_data():
    results = pd.read_csv("https://drive.google.com/uc?id=1Y8oD8HQLnXzzQIaJkJBwvUN7Ds9E4OCx&export=download")
    qualifying = pd.read_csv("https://drive.google.com/uc?id=1mcakLSYRJvoq-Am7Cpo8Be-qFD6bYl7M&export=download")
    return results, qualifying

results, qualifying = load_data()

st.title("🏁 Formula 1 Data Exploration")
st.markdown("Explore different aspects of F1 race data using graphs and statistics.")

# Sidebar
option = st.sidebar.selectbox("Choose Analysis", (
    "Q1 Lap Time Distribution",
    "Grid Start vs. Final Position",
    "Position Change by Grid Group",
    "Final Position vs. Points",
    "Fastest Lap Rank vs. Final Position",
    "Top Driver Performance"
))

if option == "Q1 Lap Time Distribution":
    def lap_time_to_seconds(lap_time_str):
        try:
            minutes, seconds = lap_time_str.split(":")
            return float(minutes) * 60 + float(seconds)
        except:
            return None

    qualifying["q1_seconds"] = qualifying["q1"].apply(lap_time_to_seconds)
    q1_cleaned = qualifying[qualifying["q1_seconds"].notna()]

    st.subheader("Distribution of Q1 Lap Times")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(q1_cleaned["q1_seconds"], bins=30, kde=True, ax=ax, color="purple")
    ax.set_xlabel("Q1 Time (seconds)")
    ax.set_ylabel("Number of Drivers")
    st.pyplot(fig)

elif option == "Grid Start vs. Final Position":
    st.subheader("Final Race Positions by Starting Grid")
    filtered_results = results[(results["grid"] >= 1) & (results["grid"] <= 20)]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
    ax.set_yticks(np.arange(1, 21, 1))
    ax.set_xlabel("Starting Grid Position")
    ax.set_ylabel("Final Race Position")
    st.pyplot(fig)

elif option == "Position Change by Grid Group":
    st.subheader("Position Changes During Race by Grid Group")
    results["grid_group"] = results["grid"].apply(lambda x: "Top 5 grid" if x <= 5 else "Grid 6–20")
    results["position_change"] = results["grid"] - results["positionOrder"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=results, x="position_change", hue="grid_group", bins=30, kde=True, multiple="stack", palette=["green", "gray"], ax=ax)
    ax.axvline(0, color='red', linestyle='--')
    ax.set_xlabel("Position Change")
    ax.set_ylabel("Number of Drivers")
    st.pyplot(fig)

elif option == "Final Position vs. Points":
    st.subheader("Points Earned by Final Race Position")
    top20 = results[results["positionOrder"] <= 20]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=top20, x="positionOrder", y="points", palette="Blues", ax=ax)
    ax.set_xlabel("Final Race Position")
    ax.set_ylabel("Points Earned")
    st.pyplot(fig)

elif option == "Fastest Lap Rank vs. Final Position":
    st.subheader("Final Position vs. Fastest Lap Rank")
    if "rank" in results.columns and "positionOrder" in results.columns:
        lap_data = results.copy()
        lap_data = lap_data[lap_data["rank"].notna() & lap_data["positionOrder"].notna()]
        lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=lap_data, x="rank", y="positionOrder", alpha=0.6, color="purple", ax=ax)
        ax.set_xlabel("Fastest Lap Rank (1 = Fastest)")
        ax.set_ylabel("Final Race Position")
        st.pyplot(fig)
    else:
        st.warning("Rank or positionOrder columns are missing from dataset.")

elif option == "Top Driver Performance":
    st.subheader("Average Final Position of Top 6 Most Active Drivers")
    top_drivers = results["driverId"].value_counts().head(6).index
    top_driver_data = results[results["driverId"].isin(top_drivers)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=top_driver_data, x="driverId", y="positionOrder", join=False, capsize=0.2, errwidth=1.5, color="navy", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Driver ID")
    ax.set_ylabel("Average Final Position")
    st.pyplot(fig)
