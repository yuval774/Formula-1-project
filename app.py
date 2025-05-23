import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Formula 1 Analysis Dashboard", layout="wide")

# =============================================================================
# DATA
# =============================================================================
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

# Helper – convert "M:SS.sss" to float seconds

def lap_time_to_seconds(t: str):
    try:
        m, s = t.split(":")
        return float(m) * 60 + float(s)
    except:
        return np.nan

qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds)
q1_cleaned = qualifying.dropna(subset=["q1_seconds"])

filtered_results = results[results["grid"].between(1, 20)]
results["grid_group"] = results["grid"].apply(lambda x: "Top 5" if x <= 5 else "P6‑20")
results["position_change"] = results["grid"] - results["positionOrder"]

lap_data = results.query("rank.notna() & positionOrder.notna()").copy()
lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)

top_drivers = results["driverId"].value_counts().head(6).index
_top_driver_data = results[results["driverId"].isin(top_drivers)]

# =============================================================================
# SIDE BAR – IMAGE + QUICK FACTS
# =============================================================================
SIDEBAR_IMAGE_URL = "https://raw.githubusercontent.com/formula1/logos/main/f1_logo_red.png"
st.sidebar.image(SIDEBAR_IMAGE_URL, use_column_width=True)

st.sidebar.markdown("### 🧠 Did You Know?")
for fact in [
    "🏁 Pole position boosts win odds by ~40%",
    "💨 Fastest lap doesn’t guarantee a podium",
    "🔁 One‑third of overtakes happen in the first 3 laps",
    "🏙️ Monaco is the slowest but hardest GP to win",
]:
    st.sidebar.markdown(fact)

# Extra fact bank for button
FACT_BANK = [
    "🔧 Pit‑crews change 4 tyres in under 2 seconds!",
    "🐝 An F1 car can drive upside‑down at 175 km/h thanks to down‑force.",
    "🚀 Brakes generate 5‑6 G – like a fighter jet landing on a carrier.",
    "🌡️ Brake discs glow at 1 000 °C under heavy braking.",
    "🎧 2005 V10s revved to 20 000 rpm; modern hybrids peak at 15 000 rpm.",
]
if "fact_i" not in st.session_state:
    st.session_state.fact_i = random.randrange(len(FACT_BANK))

# =============================================================================
# GRAPH RENDER HELPERS
# =============================================================================

def draw_graph(name: str):
    """Return (figure, insight string) for a given graph name."""
    if name == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(q1_cleaned, x="q1_seconds", bins=30, kde=True, color="mediumorchid", ax=ax)
        ax.set(title="Distribution of Q1 Lap Times", xlabel="Q1 Time (s)", ylabel="Drivers")
        return fig, "Most drivers lap 78‑100 s; right‑skew shows a few slow outliers."

    if name == "Grid Start vs Final Position":
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
        ax.set(title="Finishing Position by Starting Grid", ylabel="Finish Position")
        ax.set_yticks(np.arange(1, 21, 1))
        return fig, "Front‑row starters finish higher on average; back‑markers vary widely."

    if name == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(results, x="position_change", hue="grid_group", bins=30, kde=True, multiple="stack", palette=["seagreen", "slategray"], ax=ax)
        ax.axvline(0, color="red", ls="--")
        ax.set(title="Position Change (Grid − Finish)")
        return fig, "Top‑5 starters mostly hold/gain; P6‑20 drivers swing broadly."

    if name == "Final Position vs Points":
        top20 = results[results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(top20, x="positionOrder", y="points", palette="Blues", ax=ax)
        ax.set(title="Points by Finish Position", xlabel="Position", ylabel="Points")
        return fig, "Points drop sharply after P10 – mirroring F1 scoring rules."

    if name == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(lap_data, x="rank", y="positionOrder", alpha=0.6, color="mediumvioletred", ax=ax)
        ax.set(title="Fastest Lap Rank vs Finish", xlabel="Fastest Lap Rank (1 fast)", ylabel="Finish Position")
        return fig, "Fastest lap alone doesn’t secure a podium – strategy matters."

    # Top Driver Performance
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(_top_driver_data, x="driverId", y="positionOrder", join=False, capsize=0.2, errwidth=1.5, color="navy", ax=ax)
    ax.invert_yaxis()
    ax.set(title="Average Finish of Top 6 Most Active Drivers", ylabel="Average Finish Position")
    return fig, "Confidence intervals expose differences in driver consistency."

GRAPH_NAMES = [
    "Q1 Lap Time Distribution",
    "Grid Start vs Final Position",
    "Position Change by Grid Group",
    "Final Position vs Points",
    "Fastest Lap Rank vs Final Position",
    "Top Driver Performance",
]

# =============================================================================
# TABS
# =============================================================================

overview_tab, explorer_tab, compare_tab = st.tabs([
    "🏎️ Overview", "📊 Graph Explorer", "🔀 Compare Graphs"\
])

# ---------------------------- OVERVIEW --------------------------------------
with overview_tab:
    st.markdown("## 🏆 Formula 1 Performance Analysis")
    st.markdown("### _It’s lights out and away we gooo!_ 🚦")
    st.markdown("**Submitted by:** Yuval Vin  |  **Track:** Business Administration + Digital Innovation")

    m1, m2, m3 = st.columns(3)
    m1.metric("👥 Drivers", results["driverId"].nunique())
    m2.metric("🗓️ Races", results["raceId"].nunique())
    m3.metric("⚡ Fastest Q1 (s)", f"{q1_cleaned['q1_seconds'].min():.3f}")

    if st.button("💡 Enlighten me with an F1 fact"):
        idx = st.session_state.fact_i
        st.info(FACT_BANK[idx])
        st.session_state.fact_i = (idx + 1) % len(FACT_BANK)

# ----------------------- GRAPH EXPLORER ------------------------------------
with explorer_tab:
    st.markdown("### Choose a visualisation:")
    btn_cols = st.columns(3)
    if "current_graph" not in st.session_state:
        st.session_state.current_graph = GRAPH_NAMES[0]

    for i, name in enumerate(GRAPH_NAMES):
        if btn_cols[i % 3].button(name, key=f"gbtn_{i}"):
            st.session_state.current_graph = name

    fig, insight = draw_graph(st.session_state.current_graph)
    st.pyplot(fig)
    st.success(insight)

# ----------------------- COMPARE TAB ---------------------------------------
with compare_tab:
    st.markdown("### Compare two graphs side‑by‑side")
    colA, colB = st.columns(2)

    with colA:
        gA = st.selectbox("Graph A", GRAPH_NAMES, key="gA")
        figA, insightA = draw_graph(gA)
        st.pyplot(figA)

    with colB:
        gB = st.selectbox("Graph B", GRAPH_NAMES, index=1, key="gB")
        figB, insightB = draw_graph(gB)
        st.pyplot(figB)

    st.markdown("---")
    st.markdown(f"**Quick insight A:** {insightA}")
    st.markdown(f"**Quick insight B:** {insightB}")

    # Simple comparison comment
    if gA == gB:
        st.info("You selected the same graph twice. Choose different ones for a meaningful comparison.")
    else:
        st.info("Compare differences between the two visuals to spot patterns or contradictions.")
