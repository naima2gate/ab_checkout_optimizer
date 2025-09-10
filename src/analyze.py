# src/analyze.py
import numpy as np
import pandas as pd
import statsmodels.api as sm

def diff_in_means_crse(df: pd.DataFrame, y_col: str, treat_col: str, cluster_col: str):
    """
    Cluster-robust difference-in-means.
    """
    df2 = df.dropna(subset=[y_col, treat_col])
    # Ensure treatment column is numeric (0/1) for OLS
    if df2[treat_col].dtype == 'object' or df2[treat_col].dtype == 'bool':
        X = sm.add_constant(df2[treat_col].astype('category').cat.codes.values)
    else:
        X = sm.add_constant(df2[treat_col].values)

    y = df2[y_col].values
    model = sm.OLS(y, X)
    results = model.fit(cov_type='cluster', cov_kwds={'groups': df2[cluster_col].values})
    lift = results.params[1]
    se = results.bse[1]
    return {
        'lift': float(lift),
        'se': float(se),
        't': float(lift / se) if se > 0 else float('nan'),
        'p': float(results.pvalues[1]),
        'ci_low': float(lift - 1.96 * se),
        'ci_high': float(lift + 1.96 * se)
    }

def cuped_transform(y: np.ndarray, x: np.ndarray):
    """
    CUPED variance reduction.
    """
    cov = np.cov(y, x, ddof=1)[0,1]
    varx = np.var(x, ddof=1)
    
    # Correctly handle zero variance in covariate
    theta = 0.0
    if varx > 1e-9: # Use a small tolerance
        theta = cov / varx
    
    y_cuped = y - theta * x
    return y_cuped, theta

def summarize_lift(df: pd.DataFrame, outcome_col: str, treat_col: str, cluster_col: str, pre_cov: str = None):
    """
    Wrapper to compute lift with optional CUPED adjustment.
    """
    df2 = df.copy()
    theta = None
    if pre_cov is not None:
        # Check for presence of pre_cov column
        if pre_cov not in df2.columns:
            raise ValueError(f"Pre-experiment covariate '{pre_cov}' not found in DataFrame.")
            
        # CUPED transformation should be done on full dataset
        y = df2[outcome_col].values
        x = df2[pre_cov].values
        y_cuped, theta = cuped_transform(y, x)
        df2['y_cuped'] = y_cuped
        outcome_col_use = 'y_cuped'
    else:
        outcome_col_use = outcome_col
    
    result = diff_in_means_crse(df2, outcome_col_use, treat_col, cluster_col)
    if theta is not None:
        result['theta'] = float(theta)
    return result

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    users = pd.DataFrame({
        'user_id': [f'u{i}' for i in range(1,21)],
        'variant': np.random.choice(['control','treatment'], size=20),
        'gpv': np.random.normal(100,10,20),
        'past_7d_gpv': np.random.normal(90,15,20)
    })
    # With CUPED
    res_cuped = summarize_lift(users, 'gpv', 'variant', 'user_id', pre_cov='past_7d_gpv')
    print("Lift with CUPED:", res_cuped)
    # Without CUPED
    res_no_cuped = summarize_lift(users, 'gpv', 'variant', 'user_id')
    print("Lift without CUPED:", res_no_cuped)