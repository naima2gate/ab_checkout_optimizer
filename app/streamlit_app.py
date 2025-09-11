import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

# --- Dashboard Configuration ---
st.set_page_config(
    page_title="Checkout Optimizer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading (with error handling) ---
@st.cache_data
def load_data():
    """Loads all necessary data files."""
    data = {}
    try:
        data['exec_summary'] = pd.read_csv('data/executive_summary.csv')
        data['seq_results'] = pd.read_csv('data/sequential_results.csv')
        data['uplift_results'] = pd.read_csv('data/uplift_results.csv')
        with open('data/bandits_report.json') as f:
            data['bandits_data'] = json.load(f)
    except FileNotFoundError as e:
        st.error(f"Error: Missing data file. Please ensure '{e.filename}' exists in the 'data/' folder.")
        st.stop()
    return data

data = load_data()

# --- Title and Introduction ---
st.title("Checkout Optimizer: A/B Test & Bandits Dashboard")
st.markdown(
    """
    Welcome to the interactive analysis dashboard for the Checkout Optimizer experiment. 
    Use the controls below to explore the results of the A/B test, heterogeneous effects, 
    and multi-armed bandit simulation.
    """
)

# --- Main Dashboard Sections ---

# --- 1. Executive Summary & Key Metrics ---
st.header("1. Executive Summary")

# Extract key metrics for st.metric display
exec_df = data['exec_summary'].set_index('Metric')
lift_cuped = exec_df.loc['Lift (CUPED)']['Value']
p_lift_gt_0 = exec_df.loc['P(lift>0)']['Value']
stop_look = exec_df.loc['Sequential stop look']['Value']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="CUPED Lift on GPV",
        value=f"{lift_cuped:.2f}%",
        delta=f"{lift_cuped:.2f}% (Significant)" if p_lift_gt_0 > 0.95 else None,
        delta_color="normal"
    )
with col2:
    st.metric(
        label="Probability of Lift > 0",
        value=f"{p_lift_gt_0:.2%}",
        help="Bayesian probability that the treatment is better than control."
    )
with col3:
    st.metric(
        label="Experiment Stopped",
        value=f"Day {int(stop_look)}",
        help="The day the sequential analysis indicated a conclusive result."
    )

st.dataframe(exec_df.T, use_container_width=True)

# --- 2. Sequential Monitoring ---
st.header("2. Sequential Monitoring")
st.markdown(
    """
    This chart shows the daily lift and its statistical significance boundary. The experiment could
    be safely stopped as soon as the blue line (p-value) crosses below the orange line (alpha boundary).
    """
)

seq_df = data['seq_results']
fig_seq = px.line(seq_df, x='look', y=['p', 'alpha_boundary'],
                  labels={'value': 'P-Value', 'variable': 'Metric', 'look': 'Day'},
                  title="Sequential P-Value vs. O'Brien-Fleming Boundary")
fig_seq.add_vline(x=stop_look, line_dash="dash", line_color="green", annotation_text="Stopping Point", annotation_position="bottom")
fig_seq.update_traces(hovertemplate='Day: %{x}<br>P-Value: %{y:.4f}<extra></extra>', selector=dict(name='p'))
fig_seq.update_traces(hovertemplate='Day: %{x}<br>Boundary: %{y:.4f}<extra></extra>', selector=dict(name='alpha_boundary'))
st.plotly_chart(fig_seq, use_container_width=True)

# --- 3. Heterogeneous Treatment Effects (Uplift) ---
st.header("3. Heterogeneous Treatment Effects")
st.markdown(
    """
    The distribution of the Conditional Average Treatment Effect (CATE) shows how the feature impacted 
    individual users differently. A positive CATE indicates a user who responded well to the treatment.
    """
)
uplift_df = data['uplift_results']
uplift_learners = ['cate_t', 'cate_x']
selected_learners = st.multiselect(
    "Select Uplift Learner(s)",
    options=uplift_learners,
    default=uplift_learners
)

if selected_learners:
    fig_uplift = go.Figure()
    for learner in selected_learners:
        fig_uplift.add_trace(go.Histogram(
            x=uplift_df[learner],
            name=f"Distribution of {learner.replace('cate_', '').upper()}-Learner CATE",
            opacity=0.7,
            xbins=dict(start=uplift_df[learner].min(), end=uplift_df[learner].max(), size=(uplift_df[learner].max() - uplift_df[learner].min()) / 50)
        ))
    fig_uplift.update_layout(
        title="Distribution of Conditional Average Treatment Effect (CATE)",
        xaxis_title="CATE (User-Level GPV Lift)",
        yaxis_title="Number of Users",
        barmode='overlay',
        legend_title="Learner"
    )
    st.plotly_chart(fig_uplift, use_container_width=True)

# --- 4. Multi-Armed Bandit Simulation ---
st.header("4. Multi-Armed Bandit Simulation")
st.markdown(
    """
    This section compares different bandit policies (Thompson Sampling, UCB1, Epsilon-Greedy)
    on their ability to find the best-performing arm (treatment).
    """
)
bandits_data = data['bandits_data']

# Create columns for the metrics and the chart
bandit_col1, bandit_col2 = st.columns([1, 2])

with bandit_col1:
    policy_options = list(bandits_data['results'].keys())
    selected_policy = st.selectbox("Select Bandit Policy", policy_options, help="Choose a policy to view its performance.")
    
    res = bandits_data['results'][selected_policy]
    
    st.metric(
        label="Cumulative Reward",
        value=f"${res['cumulative_reward']:.2f}",
        help="The total reward accumulated by the policy over 1000 steps."
    )
    
    st.json({"Arm Allocation": {f"Arm {i}": alloc for i, alloc in enumerate(res['allocation'])}})
    st.metric(
        label="Probability of Correct Selection (PCS)",
        value=f"{bandits_data['pcs'][selected_policy]:.2%}",
        help="The probability of the policy identifying and exploiting the best arm."
    )

with bandit_col2:
    fig_bandit = px.line(
        x=list(range(len(res['reward_trace']))),
        y=res['reward_trace'],
        labels={'x': 'Steps', 'y': 'Cumulative Reward'},
        title=f"Cumulative Reward Over Time for {selected_policy.replace('_', ' ').title()} Policy"
    )
    st.plotly_chart(fig_bandit, use_container_width=True)
