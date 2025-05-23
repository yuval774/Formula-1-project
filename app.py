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
    qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lambda x: float(x.split(":"[0]))*60 + float(x.split(":"[1])) if ":" in x else np.nan)
    q1_cleaned = qualifying[qualifying["q1_seconds"].notna()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Drivers", len(results["driverId"].unique()))
    col2.metric("Total Races", len(results["raceId"].unique()))
    col3.metric("Fastest Q1 Time (s)", round(q1_cleaned["q1_seconds"].min(), 2))

# --- Tab 2: Graphs ---
with graphs_tab:
    option = st.selectbox("Choose Analysis", (
        "Q1 Lap Time Distribution",
        "Grid Start vs. Final Position",
        "Position Change by Grid Group",
        "Final Position vs. Points",
        "Fastest Lap Rank vs. Final Position",
        "Top Driver Performance",
        "Compare Driver Race History"
    ))

    if option == "Q1 Lap Time Distribution":
        st.subheader("Distribution of Q1 Lap Times")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(q1_cleaned["q1_seconds"], bins=30, kde=True, ax=ax, color="purple")
        ax.set_xlabel("Q1 Time (seconds)")
        ax.set_ylabel("Number of Drivers")
        st.pyplot(fig)
        with st.expander("What this graph tells us"):
            st.markdown("Most drivers clock lap times between 78–100 seconds. The slight right skew indicates a few slower performances.")

    elif option == "Grid Start vs. Final Position":
        st.subheader("Final Race Positions by Starting Grid")
        filtered_results = results[(results["grid"] >= 1) & (results["grid"] <= 20)]
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
        ax.set_yticks(np.arange(1, 21, 1))
        ax.set_xlabel("Starting Grid Position")
        ax.set_ylabel("Final Race Position")
        st.pyplot(fig)
        with st.expander("What this graph tells us"):
            st.markdown("Drivers starting in front rows tend to finish better. Back-grid drivers show wider position variation.")

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
        with st.expander("What this graph tells us"):
            st.markdown("Top 5 starters maintain or improve position. Others show more volatility in performance.")

    elif option == "Final Position vs. Points":
        st.subheader("Points Earned by Final Race Position")
        top20 = results[results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=top20, x="positionOrder", y="points", palette="Blues", ax=ax)
        ax.set_xlabel("Final Race Position")
        ax.set_ylabel("Points Earned")
        st.pyplot(fig)
        with st.expander("What this graph tells us"):
            st.markdown("Top 10 finishers earn consistent points. 1st place drivers typically score the most.")

    elif option == "Fastest Lap Rank vs. Final Position":
        st.subheader("Final Position vs. Fastest Lap Rank")
        lap_data = results.copy()
        lap_data = lap_data[lap_data["rank"].notna() & lap_data["positionOrder"].notna()]
        lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=lap_data, x="rank", y="positionOrder", alpha=0.6, color="purple", ax=ax)
        ax.set_xlabel("Fastest Lap Rank (1 = Fastest)")
        ax.set_ylabel("Final Race Position")
        st.pyplot(fig)
        with st.expander("What this graph tells us"):
            st.markdown("Fastest lap doesn't guarantee a win — speed isn't everything in F1.")

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
        with st.expander("What this graph tells us"):
            st.markdown("Driver 1 shows best average, while others vary more in performance.")

    elif option == "Compare Driver Race History":
        st.subheader("Driver Performance Over Time")
        driver_selected = st.selectbox("Select a driver to compare", results["driverId"].unique())
        driver_data = results[results["driverId"] == driver_selected].sort_values("raceId")
        if not driver_data.empty:
            st.line_chart(driver_data.set_index("raceId")["positionOrder"])
        else:
            st.info("No data found for selected driver.")

    st.markdown("---")
    st.subheader("💬 Ask ChatGPT about the data")
    user_question = st.text_input("Ask a question about the graph:", placeholder="e.g. What does this tell us about race consistency?")
    if st.button("Ask GPT") and user_question:
        with st.spinner("Thinking..."):
            response = ask_gpt(user_question)
            st.success(response)
