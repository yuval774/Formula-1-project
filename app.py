qualifying["q1_seconds"] = qualifying["q1"].dropna().apply(lap_time_to_seconds_safe)
q1_cleaned = qualifying[qualifying["q1_seconds"].notna()]
filtered_results = results[(results["grid"] >= 1) & (results["grid"] <= 20)]
results["grid_group"] = results["grid"].apply(lambda x: "Top 5 grid" if x <= 5 else "Grid 6–20")
results["position_change"] = results["grid"] - results["positionOrder"]
lap_data = results.copy()
lap_data = lap_data[lap_data["rank"].notna() & lap_data["positionOrder"].notna()]
lap_data["positionOrder"] = lap_data["positionOrder"].astype(int)
top_drivers = results["driverId"].value_counts().head(6).index
top_driver_data = results[results["driverId"].isin(top_drivers)]

# Sidebar fun facts
st.sidebar.markdown("""
### 🧠 Did You Know?
- A pole position increases win chance by 40%
- Fastest lap doesn’t guarantee a podium
- Most races are decided in the first 5 laps
""")

# Tabs layout
tabs = st.tabs(["🏁 Overview", "📊 Visualizations"])

with tabs[0]:
    st.title("Formula 1 Performance Analysis")
    st.markdown("#### It’s lights out and away we goooo!")
    st.markdown("""
    **Submitted by:** Yuval Vin  
    **Track:** Business Administration with Digital Innovation  

    This project explores the impact of qualifying positions, fastest laps, and race strategy on final results in Formula 1.  
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Drivers", len(results["driverId"].unique()))
    col2.metric("Total Races", len(results["raceId"].unique()))
    col3.metric("Fastest Q1 Time (s)", round(q1_cleaned["q1_seconds"].min(), 2))

with tabs[1]:
    st.subheader("Choose a graph to view:")
    graph_choice = st.radio("Select a graph:", [
        "Q1 Lap Time Distribution",
        "Grid Start vs Final Position",
        "Position Change by Grid Group",
        "Final Position vs Points",
        "Fastest Lap Rank vs Final Position",
        "Top Driver Performance"
    ])

    if graph_choice == "Q1 Lap Time Distribution":
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(data=q1_cleaned, x="q1_seconds", bins=30, kde=True, color="purple", ax=ax)
        ax.set_title("Distribution of Q1 Lap Times (in seconds)")
        ax.set_xlabel("Q1 Time (seconds)")
        ax.set_ylabel("Number of Drivers")
        st.pyplot(fig)
        st.markdown("**Insight:** Most drivers set lap times between 78–100 seconds. A slight right skew indicates some outliers with slower times.")

    elif graph_choice == "Grid Start vs Final Position":
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=filtered_results, x="grid", y="positionOrder", palette="pastel", ax=ax)
        ax.set_yticks(np.arange(1, 21, 1))
        ax.set_title("Final Race Positions by Starting Grid Position (1–20 Only)")
        st.pyplot(fig)
        st.markdown("**Insight:** Front-grid starters tend to finish better with tighter distribution; back-grid drivers are more variable.")

    elif graph_choice == "Position Change by Grid Group":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=results, x="position_change", hue="grid_group", bins=30, kde=True, multiple="stack", palette=["green", "gray"], ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title("Position Change Distribution by Grid Group")
        st.pyplot(fig)
        st.markdown("**Insight:** Top 5 starters usually maintain/improve position. Grid 6–20 drivers show more varied outcomes.")

    elif graph_choice == "Final Position vs Points":
        top20 = results[results["positionOrder"] <= 20]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=top20, x="positionOrder", y="points", palette="Blues", ax=ax)
        ax.set_title("Points Distribution by Final Race Position")
        st.pyplot(fig)
        st.markdown("**Insight:** Points drop off sharply after position 10, confirming F1’s reward structure for top finishers.")

    elif graph_choice == "Fastest Lap Rank vs Final Position":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=lap_data, x="rank", y="positionOrder", alpha=0.6, color="purple", ax=ax)
        ax.set_title("Final Race Position vs. Fastest Lap Rank")
        st.pyplot(fig)
        st.markdown("**Insight:** Setting the fastest lap doesn’t guarantee a top finish – speed alone isn’t enough.")

    elif graph_choice == "Top Driver Performance":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.pointplot(data=top_driver_data, x="driverId", y="positionOrder", join=False, capsize=0.2, errwidth=1.5, color="navy", ax=ax)
        ax.invert_yaxis()
        ax.set_title("Average Final Race Position of Top 6 Most Active Drivers")
        st.pyplot(fig)
        st.markdown("**Insight:** Some drivers are consistently better – confidence intervals help show reliability over time.")

    user_question = st.text_input("Ask ChatGPT about this graph:")
    if st.button("Ask GPT") and user_question:
        st.markdown(ask_gpt(user_question))
