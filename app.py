import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Formula 1 Project", layout="centered")

# App title
st.title("🏎️ Formula 1 Analysis Dashboard")

# Load results data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/YuvalVin/F1_Midterm_Project/main/data/results.csv"
    return pd.read_csv(url)

# Load the dataset
results = load_data()

# Optional: show raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw Data Preview")
    st.write(results)

# Section 1: Average points per race (trend over time)
st.subheader("📈 Average Points Per Race Over Time")
race_trend = (
    results.groupby("raceId", as_index=False)["points"]
    .mean()
    .sort_values("raceId")
)

fig1, ax1 = plt.subplots()
sns.lineplot(data=race_trend, x="raceId", y="points", ax=ax1)
ax1.set_title("Average Points Per Race")
ax1.set_xlabel("Race ID")
ax1.set_ylabel("Average Points")
st.pyplot(fig1)

# Section 2: Top 10 Drivers by Total Points
st.subheader("🏁 Top 10 Drivers by Total Points")
driver_points = (
    results.groupby("driverId", as_index=False)["points"]
    .sum()
    .sort_values("points", ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots()
sns.barplot(data=driver_points, x="driverId", y="points", palette="viridis", ax=ax2)
ax2.set_title("Top 10 Drivers - Total Points")
ax2.set_xlabel("Driver ID")
ax2.set_ylabel("Total Points")
st.pyplot(fig2)

# Section 3: Points Distribution
st.subheader("📊 Points Distribution Histogram")
fig3, ax3 = plt.subplots()
sns.histplot(results["points"], bins=30, kde=True, ax=ax3)
ax3.set_title("Distribution of Points")
ax3.set_xlabel("Points")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)

# Section 4: Boxplot of points per position
st.subheader("🎯 Boxplot: Points by Finishing Position")
fig4, ax4 = plt.subplots()
sns.boxplot(data=results, x="positionOrder", y="points", ax=ax4)
ax4.set_title("Points by Position Order")
ax4.set_xlabel("Finishing Position")
ax4.set_ylabel("Points")
st.pyplot(fig4)

# Section 5: Scatter plot - Grid vs Position
st.subheader("⚙️ Scatter Plot: Grid Start vs Final Position")
fig5, ax5 = plt.subplots()
sns.scatterplot(data=results, x="grid", y="positionOrder", hue="positionOrder", palette="cool", ax=ax5)
ax5.set_title("Starting Grid vs Final Position")
ax5.set_xlabel("Starting Grid Position")
ax5.set_ylabel("Final Position Order")
st.pyplot(fig5)

# Summary section
st.subheader("🧠 Insights & Conclusions")
st.markdown("""
- **Race difficulty** appears to increase over time, based on lower average points.
- **Top 10 drivers** significantly outperform the rest.
- **Points distribution** is skewed — most drivers score low, with few scoring high.
- **Position strongly affects points**, as shown by the boxplot.
- **Scatter plot** shows that drivers don't always finish where they started — lots of overtakes!
""")
