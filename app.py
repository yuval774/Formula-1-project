import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import openai

# ────────────────────────────────────────────────────────────────────────────────
# ▶️  CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Formula 1 Analysis Dashboard", layout="wide")

# OpenAI client (initialise only if key exists so app doesn't crash)
api_key = st.secrets.get("openai_api_key")
client = openai.OpenAI(api_key=api_key) if api_key else None

# ────────────────────────────────────────────────────────────────────────────────
# 🧠  GPT WRAPPER
# ────────────────────────────────────────────────────────────────────────────────
def ask_gpt(prompt: str) -> str:
    """Return ChatGPT answer or a clear error message."""
    if client is None:
        return "❌ Error: OpenAI API key missing – add it in Streamlit > Settings > Secrets."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {e}"

# ────────────────────────────────────────────────────────────────────────────────
# 📥 DATA LOADING & PREP
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    results = pd.read_csv(
        "https://drive.google.com/uc?id=1Y8oD8HQLnXzzQIaJkJBwvUN7Ds9E4OCx&export=download"
    )
    qualifying = pd.read_csv(
        "https://drive.google.com/uc?id=1mcakLSYRJvoq-Am7Cpo8Be-qFD6bYl7M&export=download"
    )
    return results, qualifying

results, qualifying = load_data()

# Helper: convert lap time “M:SS.mmm” ➜ seconds
def lap_time_to_seconds(t: str):
    try:
        m, s = t.split(":")
        return float(m) * 60 + float(s)
    except:
        return np.nan

qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds)
q1_cleaned = qualifying.dropna(subset=["q1_seconds"])

filtered_results = results[(results["grid"].between(1, 20))]
results["grid_group"] = results["grid"].apply(lambda x: "Top 5 grid" if x <= 5 else "Grid 6–20")
results["position_change"] = results["grid"] - results["positionOrder"]

lap_data = results.query("rank.notna() & positionOrder.notna()")
lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)

top_drivers = results["driverId"].value_counts().head(6).index
top_driver_data = results[results["driverId"].isin(top_drivers)]

# ────────────────────────────────────────────────────────────────────────────────
# 📑 SIDEBAR – FUN FACTS
# ────────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
### 🧠 Did You Know?
* Pole position boosts win probability by ~40 %
* Fastest lap ≠ guaranteed podium
* ⅓ of overtakes occur in first 3 laps
* Monaco is the slowest F1 GP but hardest to win
""")

# ────────────────────────────────────────────────────────────────────────────────
# 🖥️ LAYOUT
# ────────────────────────────────────────────────────────────────────────────────
overview_tab, viz_tab = st.tabs(["🏁 Overview", "📊 Visualisations"])

# -- Overview -------------------------------------------------------------------
with overview_tab:
    st.title("Formula 1 Performance Analysis")
    st.subheader("It’s lights out and away we goooo!")

    st.markdown(
        "**Submitted by:** Yuval Vin  \|  **Track:** Business Administration + Digital Innovation"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Drivers", f"{results['driverId'].nunique()}")
    col2.metric("Races", f"{results['raceId'].nunique()}")
    col3.metric("Fastest Q1 (s)", f"{q1_cleaned['q1_seconds'].min():.3f}")

# -- Visualisations -------------------------------------------------------------
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

    # ---- Graph 1 --------------------------------------------------------------
    if graph == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(q1_cleaned, x="q1_seconds", bins=30, kde=True, color="purple", ax=ax)
        ax.set(title="Distribution of Q1 Lap Times", xlabel="Q1 Time (s)", ylabel="Drivers")
        st.pyplot(fig)
        st.info("Most drivers lap between 78–100 seconds; a right‑skew hints at a few slow outliers.")

    # ---- Graph 2 --------------------------------------------------------------
    elif graph == "Grid Start vs Final Position":
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
        ax.set(title="Final Position by Starting Grid", ylabel="Finishing Position")
        st.pyplot(fig)
        st.info("Front‑row starters finish higher on average; back‑markers show wider variance.")

    # ---- Graph 3 --------------------------------------------------------------
    elif graph == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(
            results,
            x="position_change",
            hue="grid_group",
            bins=30,
            kde=True,
            multiple="stack",
            palette=["green", "gray"],
            ax=ax,
        )
        ax.axvline(0, color="red", ls="--")
        ax.set(title="Position Change Distribution", xlabel="Grid − Final Position")
        st.pyplot(fig)
        st.info("Top‑5 starters mostly hold station; others experience greater swings – both gains & losses.")

    # ---- Graph 4 --------------------------------------------------------------
    elif graph == "Final Position vs Points":
        top20 = results[results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=top20, x="positionOrder", y="points", palette="Blues", ax=ax)
        ax.set(title="Points by Finishing Position", xlabel="Position", ylabel="Points")
        st.pyplot(fig)
        st.info("Points collapse after P10 – reflecting F1’s scoring system rewarding top finishers.")

    # ---- Graph 5 --------------------------------------------------------------
    elif graph == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(lap_data, x="rank", y="positionOrder", alpha=0.6, color="purple", ax=ax)
        ax.set(title="Fastest Lap Rank vs Finish", xlabel="Fastest Lap Rank (1 = quickest)", ylabel="Finish Position")
        st.pyplot(fig)
        st.info("Fastest lap isn’t a podium guarantee – strategy & consistency trump raw speed.")

    # ---- Graph 6 --------------------------------------------------------------
    else:  # Top Driver Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pointplot(
            data=top_driver_data,
            x="driverId",
            y="positionOrder",
            join=False,
            capsize=0.2,
            errwidth=1.5,
            color="navy",
            ax=ax,
        )
        ax.invert_yaxis()
        ax.set(title="Avg Finish of Top 6 Most Active Drivers", ylabel="Avg Finish Position")
        st.pyplot(fig)
        st.info("Some drivers show remarkable consistency; confidence intervals reveal reliability.")

    # ── ChatGPT interaction ────────────────────────────────────────────────────
    question = st.text_input("Ask ChatGPT about this graph:")
    if st.button("Ask GPT") and question:
        st.markdown(ask_gpt(question))
