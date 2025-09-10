# src/simulate.py
import numpy as np
import pandas as pd

def generate_users(n_users=1000, seed=42):
    """
    Generate synthetic users with features.
    """
    np.random.seed(seed)
    users = pd.DataFrame({
        'user_id': [f'u{i}' for i in range(1, n_users+1)],
        'country': np.random.choice(['US','IN','UK','DE'], n_users),
        'device': np.random.choice(['mobile','desktop'], n_users),
        'traffic_source': np.random.choice(['organic','paid'], n_users),
        'past_7d_gpv': np.random.gamma(100,1,n_users)
    })
    return users

def simulate_funnel(users: pd.DataFrame, lift=0.025, heterogeneity=True, noncompliance=0.05,
                    logging_loss=0.02, seed=42):
    """
    Simulate funnel for control/treatment with optional heterogeneity and noise.
    """
    np.random.seed(seed)
    sessions = []
    events = []
    orders = []
    perf = []
    
    for _, user in users.iterrows():
        # Use the pre-assigned variant from the input DataFrame
        variant = user['variant']
        user_lift = lift
        if heterogeneity:
            # Heterogeneous lift by device/country
            if user['device']=='mobile':
                user_lift *= 1.5
            if user['country']=='IN':
                user_lift *= 0.8
        
        if np.random.rand() < noncompliance:
            # Random noncompliance flips assignment
            variant = 'control' if variant=='treatment' else 'treatment'

        # Simulate sessions (1-3 per user)
        n_sess = np.random.randint(1,4)
        for s in range(n_sess):
            session_id = f"{user['user_id']}_s{s}"
            session_day = np.random.randint(1,8) # 7 day experiment
            sessions.append({'session_id': session_id, 'user_id': user['user_id'], 'variant': variant, 'session_day': session_day})
            
            # Simulate conversion based on variant & lift
            conversion_prob = 0.10 + (user_lift if variant=='treatment' else 0)
            placed_order = np.random.rand() < conversion_prob
            
            if placed_order:
                perf.append({'session_id': session_id, 'checkout_latency_ms': np.random.normal(160,20) if variant=='control' else np.random.normal(150,15)})
            else:
                perf.append({'session_id': session_id, 'checkout_latency_ms': np.random.normal(160,20)})
            
            # Simulate events
            events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'view_page'})
            if np.random.rand() < 0.9:
                events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'add_to_cart'})
            if np.random.rand() < 0.85:
                events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'view_cart'})
            if np.random.rand() < 0.8:
                events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'add_payment'})

            if placed_order:
                events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'place_order'})
                orders.append({
                    'order_id': f"{session_id}_o1",
                    'session_id': session_id,
                    'revenue': np.random.normal(100,10),
                    'discount': np.random.normal(5,2),
                    'var_cost': np.random.normal(20,5)
                })
                if np.random.rand() < 0.05:
                    events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'refund'})
            
            if np.random.rand() < 0.02:
                events.append({'session_id': session_id, 'user_id': user['user_id'], 'name':'support_ticket'})
    
    sessions_df = pd.DataFrame(sessions)
    events_df = pd.DataFrame(events)
    orders_df = pd.DataFrame(orders)
    perf_df = pd.DataFrame(perf)
    
    return sessions_df, events_df, orders_df, perf_df

if __name__ == "__main__":
    users = generate_users()
    sessions, events, orders, perf = simulate_funnel(users, lift=0.05)
    
    print("Simulated tables:")
    print("Sessions:", sessions.shape)
    print("Events:", events.shape)
    print("Orders:", orders.shape)
    print("Perf:", perf.shape)