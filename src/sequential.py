# src/sequential.py
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.analyze import diff_in_means_crse

def o_brien_fleming_alpha(t, max_looks, alpha=0.05):
    """
    O'Brien-Fleming alpha spending function.
    """
    if t > max_looks:
        raise ValueError("Current look exceeds max looks")
    t_frac = t / max_looks
    boundary = 2 * (1 - norm.cdf(norm.ppf(1 - alpha/2) / np.sqrt(t_frac)))
    return boundary

def sequential_p_values(lift_series: pd.Series, se_series: pd.Series, max_looks: int, alpha=0.05):
    """
    Compute sequential p-values for daily or periodic looks.
    """
    results = []
    for t, (lift, se) in enumerate(zip(lift_series, se_series), start=1):
        z = lift / se if se > 0 else np.nan
        p = 2 * (1 - norm.cdf(abs(z)))
        alpha_boundary = o_brien_fleming_alpha(t, max_looks, alpha)
        stop = p < alpha_boundary
        results.append({'look': t, 'lift': lift, 'se': se, 'z': z, 'p': p,
                        'alpha_boundary': alpha_boundary, 'stop': stop})
    return pd.DataFrame(results)

def sequential_monitoring(df: pd.DataFrame, outcome_col: str, treat_col: str, cluster_col: str,
                          max_looks: int = 14, alpha: float = 0.05):
    """
    Perform sequential monitoring using cluster-robust SE.
    """
    lift_list = []
    se_list = []
    
    # Ensure 'day' column exists and is sorted
    if 'day' not in df.columns:
        raise ValueError("DataFrame must contain a 'day' column for sequential analysis.")
    
    # Get unique sorted days
    days = sorted(df['day'].unique())
    if len(days) < max_looks:
        print(f"Warning: Number of unique days ({len(days)}) is less than max_looks ({max_looks}). Using available days.")
    
    for look in range(1, min(max_looks + 1, len(days) + 1)):
        sub_df = df[df['day'] <= days[look-1]]
        res = diff_in_means_crse(sub_df, outcome_col, treat_col, cluster_col)
        lift_list.append(res['lift'])
        se_list.append(res['se'])
    
    seq_df = sequential_p_values(pd.Series(lift_list), pd.Series(se_list), max_looks, alpha)
    return seq_df

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    df = pd.DataFrame({
        'user_id': np.tile([f'u{i}' for i in range(1,21)], 7),
        'day': np.repeat(np.arange(1,8), 20),
        'variant': np.random.choice(['control','treatment'], 140),
        'gpv_cuped': np.random.normal(100,10,140)
    })
    
    seq_res = sequential_monitoring(df, 'gpv_cuped', 'variant', 'user_id', max_looks=7, alpha=0.05)
    print(seq_res)