# scripts/run_bandits.py
"""
Simulate Bandit policies and generate bandits_report.json
"""

import numpy as np
import json
import os
from src.bandits import ThompsonBernoulli, ThompsonGaussian, UCB1, EpsilonGreedy

np.random.seed(42)


# Define bandit policies
policies = {
    'epsilon_greedy': EpsilonGreedy(n_arms=2, epsilon=0.1),
    'ucb1': UCB1(n_arms=2),
    'thompson_sampling': ThompsonGaussian(n_arms=2, mu0=1.0, sigma0=1.0)
}
n_steps = 1000

bandits_results = {'results': {}, 'pcs': {}}

# True values for reward simulation
true_gpv = [100, 105]
best_arm = np.argmax(true_gpv)

for policy_name, policy in policies.items():
    cumulative_reward = 0
    reward_trace = []
    
    # Simulate rewards for each step
    for step in range(n_steps):
        arm = policy.select_arm()
        
        # Simulate reward: arm 0 = control, arm 1 = treatment
        reward = np.random.normal(true_gpv[arm], 0.1)
        
        # Update the policy with the observed reward
        policy.update(arm, reward)
        
        cumulative_reward += reward
        reward_trace.append(cumulative_reward)
    
    # Store results
    bandits_results['results'][policy_name] = {
        'cumulative_reward': round(cumulative_reward,2),
        'allocation': policy.counts.astype(int).tolist() if hasattr(policy,'counts') else policy.n.astype(int).tolist(),
        'reward_trace': reward_trace
    }
    
    # Correct PCS: check if the final most selected arm is the best arm
    if hasattr(policy,'counts'):
        chosen_arm = np.argmax(policy.counts)
    else:
        chosen_arm = np.argmax(policy.n)
    
    pcs = 1.0 if chosen_arm == best_arm else 0.0
    bandits_results['pcs'][policy_name] = pcs


# Save to JSON
os.makedirs('data', exist_ok=True)
with open('data/bandits_report.json', 'w') as f:
    json.dump(bandits_results, f, indent=4)

print("Bandit simulation complete. Report saved to 'data/bandits_report.json'")
print("Probability of Correct Selection (PCS):")
for name, val in bandits_results['pcs'].items():
    print(f"{name}: {val}")