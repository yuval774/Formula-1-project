import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Formula 1 Analysis Dashboard",
                   layout="wide")

# ─────────────────────────────────────────────────────────────
#  DATA LOAD + PREP
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔄 Loading F1 datasets…")
def load_data():
    results = pd.read_csv(
        "https://drive.google.com/uc?id=1Y8oD8HQLnXzzQIaJkJBwvUN7Ds9E4OCx&export=download"
    )
    qualifying = pd.read_csv(
        "https://drive.google.com/uc?id=1mcakLSYRJvoq-Am7Cpo8Be-qFD6bYl7M&export=download"
    )
    return results, qualifying


results, qualifying = load_data()

# Helper – convert "M:SS.sss" ➜ float seconds
def lap_time_to_seconds(t: str):
    try:
        m, s = t.split(":")
        return float(m) * 60 + float(s)
    except Exception:
        return np.nan


qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds)
q1_cleaned = qualifying.dropna(subset=["q1_seconds"])

results["grid_group"] = results["grid"].apply(
    lambda x: "Top 5" if x <= 5 else "P6-20"
)
results["position_change"] = results["grid"] - results["positionOrder"]

lap_data = results.query("rank.notna() & positionOrder.notna()").copy()
lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)

top_drivers = results["driverId"].value_counts().head(6).index
_top_driver_data = results[results["driverId"].isin(top_drivers)]

# ─────────────────────────────────────────────────────────────
#  SIDEBAR – QUICK FACTS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🧠 Did You Know?")
for quick_fact in [
    "🏁 Pole position boosts win odds by ~40%",
    "💨 Fastest lap doesn’t guarantee a podium",
    "🔁 One-third of overtakes happen in the first 3 laps",
    "🏙️ Monaco is the slowest but hardest GP to win",
]:
    st.sidebar.markdown(quick_fact)

# Fact bank for the button
FACT_BANK = [
    "🔧 Pit-crews change 4 tyres in under 2 s!",
    "🐝 An F1 car can drive upside-down at 175 km/h thanks to down-force.",
    "🚀 Brakes generate 6 G – like a fighter jet landing.",
    "🌡️ Brake discs glow at over 1 000 °C.",
    "🎧 V10 engines peaked at 20 000 rpm in 2005.",
]
if "fact_i" not in st.session_state:
    st.session_state.fact_i = random.randrange(len(FACT_BANK))

# ─────────────────────────────────────────────────────────────
#  GRAPH-DEFINITION FUNCTION
# ─────────────────────────────────────────────────────────────
def draw_graph(name: str, df_results: pd.DataFrame):
    """Return (figure, insight-string) for a given graph label."""
    if name == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.histplot(q1_cleaned, x="q1_seconds", bins=30,
                     kde=True, color="mediumorchid", ax=ax)
        ax.set(title="Distribution of Q1 Lap Times",
               xlabel="Q1 Time (s)", ylabel="Drivers")
        return fig, "Most drivers lap 78-100 s; right-skew shows slower outliers."

    if name == "Grid Start vs Final Position":
        filtered = df_results[df_results["grid"].between(1, 20)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(filtered, x="grid", y="positionOrder",
                    palette="pastel", ax=ax)
        ax.set(title="Finishing Position by Starting Grid",
               ylabel="Finish Position")
        ax.set_yticks(np.arange(1, 21, 1))
        return fig, "Front-row starters finish higher; back-markers vary widely."

    if name == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.histplot(df_results, x="position_change", hue="grid_group",
                     bins=30, kde=True, multiple="stack",
                     palette=["seagreen", "slategray"], ax=ax)
        ax.axvline(0, color="red", ls="--")
        ax.set(title="Position Change (Grid − Finish)")
        return fig, "Top-5 starters mostly hold/gain; P6-20 drivers swing broadly."

    if name == "Final Position vs Points":
        top20 = df_results[df_results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.boxplot(top20, x="positionOrder", y="points",
                    palette="Blues", ax=ax)
        ax.set(title="Points by Finish Position",
               xlabel="Position", ylabel="Points")
        return fig, "Points drop sharply after P10 – F1’s scoring rule."

    if name == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.scatterplot(lap_data, x="rank", y="positionOrder",
                        alpha=0.6, color="mediumvioletred", ax=ax)
        ax.set(title="Fastest Lap Rank vs Finish",
               xlabel="Fastest Lap Rank", ylabel="Finish Pos.")
        return fig, "Fastest lap alone doesn’t secure a podium – strategy matters."

    if name == "Top Driver Performance":
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.pointplot(_top_driver_data, x="driverId", y="positionOrder",
                      join=False, capsize=0.2, errwidth=1.5,
                      color="navy", ax=ax)
        ax.invert_yaxis()
        ax.set(title="Avg Finish – Top 6 Most Active Drivers",
               ylabel="Avg Finish Pos.")
        return fig, "Confidence intervals reveal driver consistency variations."

    # Correlation heat-map
    corr_df = df_results[["grid", "positionOrder", "points", "laps"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
    ax.set(title="Correlation Matrix – Key Metrics")
    return fig, "Shows relationships between quantitative variables."

GRAPH_NAMES = [
    "Q1 Lap Time Distribution",
    "Grid Start vs Final Position",
    "Position Change by Grid Group",
    "Final Position vs Points",
    "Fastest Lap Rank vs Final Position",
    "Top Driver Performance",
    "Correlation Heat-map (numeric)",
]

# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
overview_tab, explorer_tab, compare_tab, about_tab = st.tabs(
    ["🏎️ Overview", "📊 Graph Explorer", "🔀 Compare Graphs", "🛈 About"]
)

# ------------------------------ OVERVIEW -------------------------------------
with overview_tab:
    st.markdown("## 🏆 Formula 1 Performance Analysis")
    st.markdown("### _It’s lights out and away we gooo!_ 🚦")
    st.markdown(
        "**Submitted by:** Yuval Vin  |  **Track:** Business Administration + Digital Innovation")

    # Key metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Drivers", results["driverId"].nunique())
    c2.metric("🗓️ Races", results["raceId"].nunique())
    c3.metric("⚡ Fastest Q1 (s)",
              f"{q1_cleaned['q1_seconds'].min():.3f}")

    # Rotating fact button
    if st.button("💡 Enlighten me with an F1 fact"):
        idx = st.session_state.fact_i
        st.info(FACT_BANK[idx])
        st.session_state.fact_i = (idx + 1) % len(FACT_BANK)

    st.markdown("#### 📺 Recommended video for beginners")
    st.video("https://www.youtube.com/watch?v=Q-jjZMMxbZs")
    st.markdown(
        "_This is the **official Formula 1 YouTube channel**’s beginner guide. "
        "It explains the sport’s basics in under 10 minutes — highly recommended "
        "if you’re new to F1!_"
    )

# --------------------------- GRAPH EXPLORER ----------------------------------
with explorer_tab:
    st.markdown("### Choose a visualisation:")
    cols = st.columns(3)
    if "current_graph" not in st.session_state:
        st.session_state.current_graph = GRAPH_NAMES[0]

    for i, gname in enumerate(GRAPH_NAMES):
        if cols[i % 3].button(gname, key=f"btn_{i}"):
            st.session_state.current_graph = gname

    fig, insight = draw_graph(st.session_state.current_graph, results)
    st.pyplot(fig, use_container_width=True)
    st.success(insight)

# --------------------------- COMPARE TAB -------------------------------------
with compare_tab:
    st.markdown("### Compare two graphs side-by-side")
    colA, colB = st.columns(2)

    with colA:
        gA = st.selectbox("Graph A", GRAPH_NAMES, key="gA")
        figA, insA = draw_graph(gA, results)
        st.pyplot(figA, use_container_width=True)

    with colB:
        gB = st.selectbox("Graph B", GRAPH_NAMES, index=1, key="gB")
        figB, insB = draw_graph(gB, results)
        st.pyplot(figB, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**Insight A:** {insA}")
    st.markdown(f"**Insight B:** {insB}")

    if gA == gB:
        st.info("Select two different graphs for a meaningful comparison.")
    else:
        st.info("Observe similarities or differences between the two visuals.")

# ------------------------------ ABOUT TAB ------------------------------------
with about_tab:
    st.markdown("## 🛈 About this dashboard")
    st.markdown(
        """
**Dataset source:** “Formula-1 Race Data” on Kaggle  
<https://www.kaggle.com/datasets/jtrotman/formula-1-race-data>

**Tabs overview**

* **🏎️ Overview** – project intro, key metrics, beginner-friendly F1 video and a rotating fun-fact button  
* **📊 Graph Explorer** – large buttons to view individual insights  
* **🔀 Compare Graphs** – select any two visuals side-by-side for quick comparison  
* **🛈 About** – you are here!

Built with **Streamlit**, **Pandas**, **Seaborn**, and a passion for racing 🏎️.
"""
    )
