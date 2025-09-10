# scripts/run_ab_test.py
"""
Simulate A/B test:
1. Assign users to control/treatment (sticky, stratified)
2. Simulate exposures/events/metrics
3. Save processed tables for analysis
"""

import pandas as pd
import os
from src.assign import assign_users
from src.simulate import simulate_funnel
from src.metrics import summarize_metrics


# Load users
users = pd.read_csv('data/users.csv')
# Assign users to variants
variant_list = ['control','treatment']
assignments = assign_users(users, exp_id='checkout_optimizer', variant_list=variant_list, strata_cols=['country','device'])
assignments.to_csv('data/assignments.csv', index=False)


# Merge user data with assignments before simulation
users_with_assignments = users.merge(assignments, on='user_id', how='left')


# Simulate sessions & events using corrected simulate.py
sessions, events, orders, perf = simulate_funnel(users_with_assignments, lift=0.05)


# Save all tables
sessions.to_csv('data/sessions.csv', index=False)
events.to_csv('data/events.csv', index=False)
orders.to_csv('data/orders.csv', index=False)
perf.to_csv('data/perf.csv', index=False)

print("A/B test simulation complete. Tables saved in 'data/' folder.")