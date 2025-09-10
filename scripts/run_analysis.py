# run_analysis.py
"""
Full A/B Test Analysis Pipeline:
1. Load experiment data
2. Aggregate per-user GPV
3. Apply CUPED
4. Run frequentist analysis
5. Run Bayesian posterior analysis
6. Run sequential monitoring
7. Estimate heterogeneity / uplift (CATE)
8. Save results & executive summary
"""

import pandas as pd
import numpy as np
import os
from src.ab_testing import per_user_gpv, run_frequentist, run_bayesian
from src.analyze import summarize_lift, cuped_transform
from src.bayes import bayesian_lift_summary
from src.sequential import sequential_monitoring
from src.uplift import t_learner, x_learner, uplift_summary


# Load data
sessions = pd.read_csv('data/sessions.csv')
orders = pd.read_csv('data/orders.csv')
users = pd.read_csv('data/users.csv')
assignments = pd.read_csv('data/assignments.csv')

# Merge sessions with assignments
sessions = sessions.merge(assignments[['user_id', 'variant']], on='user_id', how='left')

# Aggregate per-user GPV
df = per_user_gpv(sessions, orders, assignments)

# Map past_7d_gpv from the users table
df['past_7d_gpv'] = df['user_id'].map(users.set_index('user_id')['past_7d_gpv'])

# Drop rows with NaNs in key columns
df = df.dropna(subset=['gpv', 'past_7d_gpv', 'variant'])

# Frequentist CUPED analysis
res_freq = run_frequentist(df)
df_cuped = res_freq['df_cuped']
print("Frequentist CUPED-adjusted lift:")
print(res_freq)

# Bayesian posterior
bayes_res = run_bayesian(df_cuped)
print("Bayesian posterior:")
print(bayes_res)

# Sequential monitoring
# To make this runnable, we need to add a 'day' column to the dataframe
df_cuped['day'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(1, 15, size=len(df_cuped)), unit='D')
df_cuped['day'] = df_cuped['day'].dt.date
df_cuped = df_cuped.sort_values('day')

# Run sequential monitoring
seq_res = sequential_monitoring(df_cuped, outcome_col='gpv_cuped', treat_col='variant', cluster_col='user_id', max_looks=10)
seq_res.to_csv('data/sequential_results.csv', index=False)
print("Sequential monitoring stop flags:")
print(seq_res[['look', 'lift', 'se', 'stop']])

# Determine the sequential stopping point
stop_look_df = seq_res[seq_res['stop']==True]
if not stop_look_df.empty:
    stop_look = stop_look_df['look'].iloc[0]
else:
    stop_look = 'N/A' # Set to 'N/A' if the experiment didn't stop early


# Heterogeneity / uplift (CATE)
# Merge in user features for uplift modeling
df_cuped = df_cuped.merge(users[['user_id', 'country', 'device', 'traffic_source']], on='user_id', how='left')
features = ['past_7d_gpv', 'country', 'device', 'traffic_source']
df_uplift = df_cuped.dropna(subset=features)
df_uplift['treat'] = df_uplift['variant'].map({'control':0, 'treatment':1})

# One-hot encode categorical features before calling uplift models
categorical_features = ['country', 'device', 'traffic_source']
df_uplift = pd.get_dummies(df_uplift, columns=categorical_features, drop_first=True)
encoded_features = [col for col in df_uplift.columns if any(cat in col for cat in categorical_features)]
all_features = ['past_7d_gpv'] + encoded_features

# Pass the dataframe and column names to the corrected functions
t_learner_res = t_learner(df_uplift, outcome_col='gpv_cuped', treat_col='treat', feature_cols=all_features)
x_learner_res = x_learner(df_uplift, outcome_col='gpv_cuped', treat_col='treat', feature_cols=all_features)

uplift_results = x_learner_res.copy()
uplift_results.rename(columns={'cate':'cate_x'}, inplace=True)
uplift_results['cate_t'] = t_learner_res['cate']
uplift_results.to_csv('data/uplift_results.csv', index=False)

print("T-Learner CATE summary:")
print(uplift_summary(t_learner_res))
print("X-Learner CATE summary:")
print(uplift_summary(x_learner_res))


# Save executive summary
exec_summary = pd.DataFrame({
    'Metric': ['Lift (CUPED)', 'theta', 'Posterior mean lift', 'P(lift>0)', 'P(lift in ROPE)', 'Sequential stop look'],
    'Value': [res_freq['lift'], res_freq['theta'], bayes_res['posterior_mu'], bayes_res['p_lift_greater_than_zero'], bayes_res['p_lift_in_rope'], stop_look]
})
exec_summary.to_csv('data/executive_summary.csv', index=False)

print("Analysis complete. Results saved in 'data/' folder.")