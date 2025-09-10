# streamlit_app.py
"""
Interactive Dashboard for:
1. A/B Test Results
2. Bandits Simulation
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

st.set_page_config(page_title="Checkout Optimizer Dashboard", layout="wide")

st.title("Checkout Optimizer: A/B Test & Bandits Dashboard")

# -----------------------------
# Load executive summary
# -----------------------------
exec_summary = pd.read_csv('data/executive_summary.csv')
st.subheader("Executive Summary")
st.dataframe(exec_summary)

# -----------------------------
# Load segment-level CATE
# -----------------------------
st.subheader("Heterogeneous Treatment Effects")

# Load cate results if available
try:
    # Load the correct uplift results file
    uplift_df = pd.read_csv('data/uplift_results.csv')
    
    st.write("User-level CATE (T-Learner and X-Learner) distribution")
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Plot both T-Learner and X-Learner CATE distributions for comparison
    ax.hist(uplift_df['cate_t'], bins=30, alpha=0.7, label='T-Learner')
    ax.hist(uplift_df['cate_x'], bins=30, alpha=0.7, label='X-Learner')
    
    ax.set_xlabel("CATE")
    ax.set_ylabel("Number of Users")
    ax.set_title("Distribution of Conditional Average Treatment Effect")
    ax.legend()
    st.pyplot(fig)

except FileNotFoundError:
    st.info("CATE data not found. Run run_analysis.py first.")

# -----------------------------
# Sequential Monitoring
# -----------------------------
st.subheader("Sequential Monitoring")
try:
    seq_df = pd.read_csv('data/sequential_results.csv')
    st.line_chart(seq_df[['look', 'lift']].set_index('look'))
except FileNotFoundError:
    st.info("Sequential monitoring data not found. Run run_analysis.py first.")

# -----------------------------
# Bandits simulation visualization
# -----------------------------
st.subheader("Bandits Simulation Results")
try:
    with open('data/bandits_report.json') as f:
        bandits_data = json.load(f)

    policy_options = list(bandits_data['results'].keys())
    selected_policy = st.selectbox("Select Bandit Policy", policy_options)

    res = bandits_data['results'][selected_policy]
    st.write(f"Cumulative Reward: {res['cumulative_reward']}")
    st.write(f"Allocation per Arm: {res['allocation']}")

    # Plot reward trace
    plt.figure(figsize=(10,4))
    plt.plot(res['reward_trace'], label="Cumulative Reward")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.title(f"{selected_policy} Reward Trace")
    plt.legend()
    st.pyplot(plt)

    # Display PCS
    pcs = bandits_data['pcs'][selected_policy]
    st.write(f"Probability of Correct Selection (PCS): {pcs}")
except FileNotFoundError:
    st.info("Bandits simulation data not found. Run run_bandits.py first.")