import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# =============================================================================
# CONFIG
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

# helper to convert Q‑times

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
# SIDEBAR – FUN FACTS + IMAGE
# =============================================================================
SIDEBAR_IMAGE_URL = "https://raw.githubusercontent.com/formula1/logos/main/f1_logo_red.png"  # placeholder logo
st.sidebar.image(SIDEBAR_IMAGE_URL, use_column_width=True)

st.sidebar.markdown("### 🧠 Did You Know?")
for fact in [
    "🏁 Pole position boosts win odds by **~40 %**",
    "💨 Fastest lap does not guarantee a podium",
    "🔁 One‑third of overtakes happen in the first 3 laps",
    "🏙️ Monaco is the slowest but hardest GP to win",
]:
    st.sidebar.markdown(fact)

# big fact bank for button
FACT_BANK = [
    "🔧 Pit‑crews change 4 tyres in < 2 s!",
    "🐝 An F1 car can drive upside‑down at 175 km/h due to downforce.",
    "🚀 Brakes pull 5‑6 G – similar to a fighter jet landing on a carrier.",
    "🌡️ Brake discs glow at 1 000 °C during heavy braking.",
    "🎧 2005 V10s revved to 20 000 rpm!",
]
if "fact_i" not in st.session_state:
    st.session_state.fact_i = random.randrange(len(FACT_BANK))

# =============================================================================
# TABS LAYOUT
# =============================================================================

overview_tab, viz_tab, compare_tab = st.tabs([
    "🏎️ Overview", "📊 Graph Explorer", "🔀 Compare Two Graphs"])

# -----------------------------------------------------------------------------
# OVERVIEW TAB
# -----------------------------------------------------------------------------
with overview_tab:
    st.markdown("## 🏆 Formula 1 Performance Analysis")
    st.markdown("### _It’s lights out and away we gooo!_ 🚦")
    st.markdown("**Submitted by:** Yuval Vin  |  **Track:** Business Administration + Digital Innovation")

    m1, m2, m3 = st.columns(3)
    m1.metric("👥 Drivers", results["driverId"].nunique())
    m2.metric("🗓️ Races", results["raceId"].nunique())
    m3.metric("⚡ Fastest Q1 (s)", f"{q1_cleaned['q1_seconds'].min():.3f}")

    if st.button("💡 Enlighten me with an F1 fact"):
        idx = st.session_state.fact_i
        st.info(FACT_BANK[idx])
        st.session_state.fact_i = (idx + 1) % len(FACT_BANK)

# -----------------------------------------------------------------------------
# GRAPH EXPLORER TAB (BIG VISUAL BUTTONS)
# -----------------------------------------------------------------------------

graph_names = [
    "Q1 Lap Time Distribution",
    "Grid Start vs Final Position",
    "Position Change by Grid Group",
    "Final Position vs Points",
    "Fastest Lap Rank vs Final Position",
    "Top Driver Performance",
]

graph_selected = graph_names[0]  # default

with viz_tab:
    st.markdown("### Choose a visualisation:")
    # Display big buttons in two‑column grid
    btn_cols = st.columns(3)
    for i, name in enumerate(graph_names):
        if btn_cols[i % 3].button(name, key=f"btn_{i}"):
            graph_selected = name
            st.session_state["current_graph"] = name
    graph_selected = st.session_state.get("current_graph", graph_selected)

    # Render chosen graph
    if graph_selected == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(q1_cleaned, x="q1_seconds", bins=30, kde=True, color="mediumorchid", ax=ax)
        ax.set(title="Distribution of Q1 Lap Times", xlabel="Q1 Time (s)", ylabel="Drivers")
        st.pyplot(fig)
        st.success("Most drivers lap 78‑100 s; right‑skew shows a few slow outliers.")

    elif graph_selected == "Grid Start vs Final Position":
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
        ax.set(title="Finishing Position by Starting Grid", ylabel="Finish Position")
        ax.set_yticks(np.arange(1, 21, 1))
        st.pyplot(fig)
        st.success("Front‑row starters finish higher on average; back‑markers vary widely.")

    elif graph_selected == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(results, x="position_change", hue="grid_group", bins=30, kde=True, multiple="stack", palette=["seagreen", "slategray"], ax=ax)
        ax.axvline(0, color="red", ls="--")
        ax.set(title="Position Change (Grid − Finish)")
        st.pyplot(fig)
        st.success("Top‑5 starters usually hold/gain; P6‑20 drivers swing broadly.")

    elif graph_selected == "Final Position vs Points":
        top20 = results[results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=top20, x="positionOrder", y="points", palette="Blues", ax=ax)
        ax.set(title="Points by Finish Position", xlabel="Position", ylabel="Points")
        st.pyplot(fig)
        st.success("Points drop sharply after P10 – reflecting F1 scoring rules.")

    elif graph_selected == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(lap_data, x="rank", y="positionOrder", alpha=0.6, color="mediumvioletred", ax=ax)
        ax.set(title="Fastest Lap Rank vs Finish", xlabel="Fastest Lap Rank (1 fast)", ylabel="Finish Position")
        st.pyplot(fig)
        st.success("Fastest lap alone doesn’t guarantee a podium – strategy matters.")

    else:  # Top Driver Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pointplot(data=_top_driver_data, x="driverId", y="positionOrder", join=False, capsize=0.2, errwidth=1.5, color="navy", ax=ax)
        ax.invert_yaxis()
        ax.set(title="Average Finish of Top 6 Most Active Drivers", ylabel="Average Finish Position")
        st.pyplot(fig)
        st.success("Confidence intervals expose differences in driver consistency.")

# -----------------------------------------------------------------------------
Skipped 1 messages
