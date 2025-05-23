import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Formula 1 Analysis Dashboard", layout="wide")

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading F1 datasets…")
def load_data():
    results = pd.read_csv(
        "https://drive.google.com/uc?id=1Y8oD8HQLnXzzQIaJkJBwvUN7Ds9E4OCx&export=download"
    )
    qualifying = pd.read_csv(
        "https://drive.google.com/uc?id=1mcakLSYRJvoq-Am7Cpo8Be-qFD6bYl7M&export=download"
    )
    return results, qualifying

results, qualifying = load_data()

# Helper: convert lap time M:SS.mmm → seconds

def lap_time_to_seconds(t: str):
    try:
        m, s = t.split(":")
        return float(m) * 60 + float(s)
    except:
        return np.nan

qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds)
q1_cleaned = qualifying.dropna(subset=["q1_seconds"])

filtered_results = results[results["grid"].between(1, 20)]
results["grid_group"] = results["grid"].apply(lambda x: "Top 5" if x <= 5 else "P6-20")
results["position_change"] = results["grid"] - results["positionOrder"]

lap_data = results.query("rank.notna() & positionOrder.notna()").copy()
lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)

top_drivers = results["driverId"].value_counts().head(6).index
_top_driver_data = results[results["driverId"].isin(top_drivers)]

# ── Fun‑fact reservoirs ───────────────────────────────────────────────────────
SIDEBAR_FACTS = [
    "🏁 Pole position boosts win odds by ~40%",
    "💨 Fastest lap does not guarantee a podium finish",
    "🔁 About one‑third of overtakes happen in the first 3 laps",
    "🏙️ Monaco is the slowest but hardest GP to win",
]

BUTTON_FACTS = [
    "🔧 F1 pit‑crews can change all 4 tyres in under 2 seconds!",
    "🐝 An F1 car can drive upside‑down in a tunnel at ~175 km/h due to its downforce.",
    "⚙️ Drivers lose up to 3 kg in sweat during a hot race.",
    "🚀 Modern F1 brakes can stop from 200 km/h to 0 in just 65 m.",
    "🌡️ Brake discs reach temperatures of **1,000 °C** during heavy braking.",
    "⛽ Since 2014, turbo‑hybrid engines achieve over 50% thermal efficiency — more than any road car.",
    "🎧 The V10 era peaked at 20,000 rpm; today’s V6s rev to 15,000 rpm but produce more torque.",
    "🪂 Drag‑reduction system (DRS) reduces rear‑wing drag by ~20% for overtaking zones.",
    "🧠 Drivers sustain lateral forces over 5 G in fast corners like Silverstone’s Copse.",
    "🔋 Teams harvest up to 4 MJ of kinetic & thermal energy per lap for extra boost.",
]

# keep track for button
if "fact_index" not in st.session_state:
    st.session_state.fact_index = random.randint(0, len(BUTTON_FACTS) - 1)

# ── Sidebar fun facts ────────────────────────────────────────────────────────
st.sidebar.markdown("### 🧠 Did You Know?")
for f in SIDEBAR_FACTS:
    st.sidebar.markdown(f)

# ── Layout ────────────────────────────────────────────────────────────────────
overview_tab, viz_tab = st.tabs(["🏎️ Overview", "📊 Visualisations"])

with overview_tab:
    st.markdown("## 🏆 Formula 1 Performance Analysis")
    st.markdown("### _It’s lights out and away we gooo!_ 🚦")
    st.markdown("**Submitted by:** Yuval Vin  |  **Track:** Business Administration + Digital Innovation")

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Drivers", f"{results['driverId'].nunique()}")
    col2.metric("🗓️ Races", f"{results['raceId'].nunique()}")
    col3.metric("⚡ Fastest Q1 (s)", f"{q1_cleaned['q1_seconds'].min():.3f}")

with viz_tab:
    graph = st.radio(
        "Select a graph:",
        [
            "Q1 Lap Time Distribution",
            "Grid Start vs Final Position",
            "Position Change by Grid Group",
            "Final Position vs Points",
            "Fastest Lap Rank vs Final Position",
            "Top Driver Performance",
        ],
        horizontal=True,
    )

    # ---- Graph 1 ------------------------------------------------------------
    if graph == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(q1_cleaned, x="q1_seconds", bins=30, kde=True, color="mediumorchid", ax=ax)
        ax.set(title="Distribution of Q1 Lap Times", xlabel="Q1 Time (s)", ylabel="Drivers")
        st.pyplot(fig)
        st.success("Most drivers lap 78‑100 s; right‑skew reveals some slow outliers.")

    # ---- Graph 2 ------------------------------------------------------------
    elif graph == "Grid Start vs Final Position":
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
        ax.set(title="Finishing Position by Starting Grid", ylabel="Finish Position")
        ax.set_yticks(np.arange(1, 21, 1))
        st.pyplot(fig)
        st.success("Front‑row starters finish higher on average; back‑markers show wider spread.")

    # ---- Graph 3 ------------------------------------------------------------
    elif graph == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            results,
            x="position_change",
            hue="grid_group",
            bins=30,
            kde=True,
            multiple="stack",
            palette=["seagreen", "slategray"],
            ax=ax,
        )
        ax.axvline(0, color="red", ls="--")
        ax.set(title="Position Change (Grid − Finish)")
        st.pyplot(fig)
        st.success("Top‑5 starters mostly hold/gain positions; P6‑20 drivers swing widely.")

    # ---- Graph 4 ------------------------------------------------------------
    elif graph == "Final Position vs Points":
        top20 = results[results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=top20, x="positionOrder", y="points", palette="Blues", ax=ax)
        ax.set(title="Points by Finish Position", xlabel="Position", ylabel="Points")
        st.pyplot(fig)
        st.success("Points drop sharply after P10 – mirroring F1’s scoring system.")

    # ---- Graph 5 ------------------------------------------------------------
    elif graph == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(lap_data, x="rank", y="positionOrder", alpha=0.6, color="mediumvioletred", ax=ax)
        ax.set(title="Fastest Lap Rank vs Finish", xlabel="Fastest Lap Rank (1 fast)", ylabel="Finish Position")
        st.pyplot(fig)
        st.success("Fastest lap alone doesn’t secure a podium – strategy matters too.")

    # ---- Graph 6 ------------------------------------------------------------
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pointplot(
            data=_top_driver_data,
            x="driverId",
            y="positionOrder",
            join=False,
            capsize=0.2,
            errwidth=1.5,
            color="navy",
            ax=ax,
        )
        ax.invert_yaxis()
        ax.set(title="Average Finish of Top 6 Most Active Drivers", ylabel="Average Finish Position")
        st.pyplot(fig)
        st.success("Confidence intervals expose differences in driver consistency.")

    # ── Random fact button ---------------------------------------------------
    if st.button("💡 Enlighten me with an F1 fact"):
        # rotate through facts without immediate repetition
        idx = st.session_state.fact_index
        st.info(BUTTON_FACTS[idx])
        st.session_state.fact_index = (idx + 1) % len(BUTTON_FACTS)
