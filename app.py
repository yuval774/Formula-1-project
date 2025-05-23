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

# --- Tab 1: Overview ---
with overview_tab:
    st.image("https://cdn-5.motorsport.com/images/mgl/0ANEnNa6/s8/sebastian-vettel-formula-one-.avif", width=640)
    st.title("Formula 1 Performance Analysis")
    st.markdown("#### It’s lights out and away we goooo!")
    st.markdown("""
    **Submitted by:** Yuval Vin  
    **Track:** Business Administration with Digital Innovation  

    This project explores the impact of qualifying positions, fastest laps, and race strategy on final results in Formula 1.  
    We examine how starting grid positions relate to finishing positions, analyze driver consistency, and investigate race volatility.
    """)
