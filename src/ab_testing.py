# src/ab_testing.py
import pandas as pd
from src.metrics import compute_gpv
from src.analyze import diff_in_means_crse, cuped_transform
from src.bayes import posterior_diff_normal, prob_greater_than_zero, prob_in_rope

def per_user_gpv(sessions: pd.DataFrame, orders: pd.DataFrame, assignments: pd.DataFrame):
    """
    Aggregate GPV per user.
    """
    if orders.empty:
        sessions = sessions.copy()
        sessions['gpv'] = 0.0
    
    gpv = compute_gpv(orders).reset_index()
    # Merge on user_id and then aggregate, as gpv is per session
    sess = sessions.merge(gpv.rename(columns={'gpv':'session_gpv'}), left_on='session_id', right_on='session_id', how='left').fillna(0.0)
    
    # Merge with assignments to get the variant info
    sess = sess.merge(assignments[['user_id', 'variant']], on='user_id', how='left')
    
    agg = sess.groupby(['user_id','variant'], as_index=False)['session_gpv'].sum()
    agg.rename(columns={'session_gpv':'gpv'}, inplace=True)
    return agg

def run_frequentist(df: pd.DataFrame):
    """
    Frequentist ITT analysis with CUPED adjustment.
    """
    df2 = df.copy()
    y = df2['gpv'].values
    x = df2['past_7d_gpv'].values
    y_cuped, theta = cuped_transform(y, x)
    df2['gpv_cuped'] = y_cuped
    
    # Correctly map variant to numeric and run diff-in-means
    df2['variant_map'] = df2['variant'].map({'control': 0, 'treatment': 1})
    res = diff_in_means_crse(df2, 'gpv_cuped', 'variant_map', 'user_id')
    res['theta'] = theta
    res['df_cuped'] = df2 # Return the modified df to prevent re-computation
    return res

def run_bayesian(df: pd.DataFrame):
    """
    Bayesian analysis on CUPED-adjusted outcomes.
    """
    df2 = df.copy()
    y_control = df2[df2['variant']=='control']['gpv_cuped'].values
    y_treat = df2[df2['variant']=='treatment']['gpv_cuped'].values
    
    # Check for empty arrays to prevent errors
    if len(y_control) < 2 or len(y_treat) < 2:
        return {'post_mean': float('nan')}
    
    posterior_mu, posterior_sigma = posterior_diff_normal(y_control, y_treat)
    p_lift_greater_than_zero = prob_greater_than_zero(posterior_mu, posterior_sigma)
    p_lift_in_rope = prob_in_rope(posterior_mu, posterior_sigma)
    
    return {
        'posterior_mu': float(posterior_mu),
        'posterior_sigma': float(posterior_sigma),
        'p_lift_greater_than_zero': float(p_lift_greater_than_zero),
        'p_lift_in_rope': float(p_lift_in_rope)
    }