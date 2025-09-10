# generate_data.py
"""
Generate synthetic data for A/B testing pipeline:
- users.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Users table
n_users = 1000
users = pd.DataFrame({
    'user_id': [f'u{i}' for i in range(n_users)],
    'country': np.random.choice(['US', 'IN', 'UK'], n_users),
    'device': np.random.choice(['desktop','mobile'], n_users),
    'traffic_source': np.random.choice(['organic','paid'], n_users),
    'past_7d_gpv': np.random.normal(100, 20, n_users)
})
users.to_csv('data/users.csv', index=False)

# -----------------------------
# Sessions, Events, Orders, Perf tables
# These are generated as part of run_ab_test.py, this file will only
# create the base users.csv file and let the next script handle the simulation.
# -----------------------------

print("Synthetic data generated successfully in 'data/' folder.")