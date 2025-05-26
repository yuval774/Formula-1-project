import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Formula 1 Analysis Dashboard", layout="wide")

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

def lap_time_to_seconds(t: str):
    try:
        m, s = t.split(":")
        return float(m) * 60 + float(s)
    except Exception:
        return np.nan


qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds)
q1_cleaned = qualifying.dropna(subset=["q1_seconds"])

# Base preprocessing (will be re-filtered later)
base_results = results.copy()
base_results["grid_group"] = base_results["grid"].apply(
    lambda x: "Top 5" if x <= 5 else "P6-20"
)
base_results["position_change"] = (
    base_results["grid"] - base_results["positionOrder"]
)

# ─────────────────────────────────────────────────────────────
#  SIDEBAR – FILTER + FACTS
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### Filter by driver (optional)")
all_drivers = sorted(base_results["driverId"].unique())
driver_opt = st.sidebar.selectbox("Choose driverId", ["All"] + all_drivers)

# Apply filter
df_view = (
    base_results
    if driver_opt == "All"
    else base_results[base_results["driverId"] == driver_opt]
)

# CSV download of the view
csv_bytes = df_view.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇️ Download filtered results CSV",
    csv_bytes,
    "f1_results_filtered.csv",
    mime="text/csv"
)

# Quick facts
st.sidebar.markdown("### 🧠 Did You Know?")
for quick_fact in [
    "🏁 Pole position boosts win odds by ~40%",
    "💨 Fastest lap doesn’t guarantee a podium",
    "🔁 One-third of overtakes happen in the first 3 laps",
    "🏙️ Monaco is the slowest but hardest GP to win",
]:
    st.sidebar.markdown(quick_fact)

# Fun-fact button state
FACT_BANK = [
    "🔧 Pit-crews change four tyres in under 2 s!",
    "🐝 An F1 car can drive upside-down at 175 km/h thanks to down-force.",
    "🚀 Brakes generate 6 G — like a fighter-jet landing.",
    "🌡️ Brake discs glow at over 1 000 °C.",
    "🎧 V10 engines peaked at 20 000 rpm in 2005.",
    "🪫 A full hybrid battery deploy adds ~160 hp for 33 s per lap.",
    "🛑 Cars decelerate from 200 km/h to 0 in ~65 m.",
    "🧊 Teams chill fuel to ~10 °C to pack more energy.",
    "📏 Minimum car weight (2024) is 798 kg incl. driver.",
    "👨‍🔬 Steering wheels cost >$50 000 each.",
    "👂 Drivers lose up to 3 kg of water in a hot race.",
    "🏎️ Power-to-weight ≈ 1 400 hp/tonne.",
    "💸 Team budget cap ≈ $135 M per season.",
    "🌍 2024 calendar = 24 races on 5 continents.",
    "⚡ DRS cuts rear-wing drag by ~20 %.",
]
if "fact_i" not in st.session_state:
    st.session_state.fact_i = 0

# ─────────────────────────────────────────────────────────────
#  GRAPH FUNCTION
# ─────────────────────────────────────────────────────────────
def draw_graph(name: str, df: pd.DataFrame):
    lap_data = df.query("rank.notna() & positionOrder.notna()").copy()
    lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)
    top_drivers = df["driverId"].value_counts().head(6).index
    top_df = df[df["driverId"].isin(top_drivers)]

    if name == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.histplot(q1_cleaned, x="q1_seconds", bins=30,
                     kde=True, color="mediumorchid", ax=ax)
        ax.set(title="Distribution of Q1 Lap Times",
               xlabel="Q1 Time (s)", ylabel="Drivers")
        return fig, "Most drivers lap 78-100 s; right-skew shows slower outliers."

    if name == "Grid Start vs Final Position":
        filtered = df[df["grid"].between(1, 20)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(filtered, x="grid", y="positionOrder",
                    palette="pastel", ax=ax)
        ax.set(title="Finishing Position by Starting Grid",
               ylabel="Finish Position")
        ax.set_yticks(np.arange(1, 21, 1))
        return fig, "Front-row starters finish higher; back-markers vary widely."

    if name == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.histplot(df, x="position_change", hue="grid_group",
                     bins=30, kde=True, multiple="stack",
                     palette=["seagreen", "slategray"], ax=ax)
        ax.axvline(0, color="red", ls="--")
        ax.set(title="Position Change (Grid − Finish)")
        return fig, "Top-5 starters mostly hold/gain; P6-20 drivers swing broadly."

    if name == "Final Position vs Points":
        top20 = df[df["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.boxplot(top20, x="positionOrder", y="points",
                    palette="Blues", ax=ax)
        ax.set(title="Points by Finish Position",
               xlabel="Position", ylabel="Points")
        return fig, "Points drop sharply after P10 — F1 scoring rule."

    if name == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(9, 3.8))
        sns.scatterplot(lap_data, x="rank", y="positionOrder",
                        alpha=0.6, color="mediumvioletred", ax=ax)
        ax.set(title="Fastest Lap Rank vs Finish",
               xlabel="Fastest Lap Rank", ylabel="Finish Pos.")
        return fig, "Fastest lap alone doesn’t secure a podium — strategy matters."

    # Top Driver Performance
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.pointplot(top_df, x="driverId", y="positionOrder",
                  join=False, capsize=0.2, errwidth=1.5,
                  color="navy", ax=ax)
    ax.invert_yaxis()
    ax.set(title="Avg Finish – Top 6 Most Active Drivers",
           ylabel="Avg Finish Pos.")
    return fig, "Confidence intervals reveal driver consistency variations."

GRAPH_NAMES = [
    "Q1 Lap Time Distribution",
    "Grid Start vs Final Position",
    "Position Change by Grid Group",
    "Final Position vs Points",
    "Fastest Lap Rank vs Final Position",
    "Top Driver Performance",
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
    st.markdown("**Submitted by:** Yuval Vin  |  **Track:** Business Administration + Digital Innovation")

    c1, c2, c3 = st.columns(3)
    c1.metric("👥 Drivers", df_view["driverId"].nunique())
    c2.metric("🗓️ Races", df_view["raceId"].nunique())
    c3.metric("⚡ Fastest Q1 (s)", f"{q1_cleaned['q1_seconds'].min():.3f}")

    if st.button("💡 Enlighten me with an F1 fact"):
        idx = st.session_state.fact_i
        if idx < len(FACT_BANK):
            st.info(FACT_BANK[idx])
            st.session_state.fact_i += 1
        else:
            st.success("You have gone through all the facts—you're really into F1 now!")
            st.session_state.fact_i = 0

    st.markdown("#### 📺 Recommended video for beginners")
    st.markdown(
        "[Watch the official F1 beginner’s guide on YouTube]"
        "(https://www.youtube.com/watch?v=Q-jjZMMxbZs)"
    )
    st.markdown(
        "_This is the **official Formula&nbsp;1 YouTube channel**’s beginner guide. "
        "It explains the sport’s basics in under 10&nbsp;minutes — highly recommended if you’re new to F1!_",
        unsafe_allow_html=True
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

    fig, insight = draw_graph(st.session_state.current_graph, df_view)
    st.pyplot(fig, use_container_width=True)
    st.success(insight)

# --------------------------- COMPARE TAB -------------------------------------
with compare_tab:
    st.markdown("### Compare two graphs side-by-side")
    colA, colB = st.columns(2)

    with colA:
        gA = st.selectbox("Graph A", GRAPH_NAMES, key="gA")
        figA, insA = draw_graph(gA, df_view)
        st.pyplot(figA, use_container_width=True)

    with colB:
        gB = st.selectbox("Graph B", GRAPH_NAMES, index=1, key="gB")
        figB, insB = draw_graph(gB, df_view)
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

**Google Colab notebook:**  
<https://colab.research.google.com/drive/1MQLCNPQfB7MmNILDuATWw_4gGCS6gTIm?usp=sharing>

**Tabs overview**

* **🏎️ Overview** – project intro, key metrics, beginner-friendly F1 video link and a rotating fun-fact button  
* **📊 Graph Explorer** – large buttons to view individual insights  
* **🔀 Compare Graphs** – select any two visuals side-by-side for quick comparison  
* **🛈 About** – you are here!

Built with **Streamlit**, **Pandas**, **Seaborn**, and a passion for racing 🏎️.
"""
    )
