# src/assign.py
import pandas as pd
import numpy as np
import hashlib

def hash_user(user_id: str, salt: str = "") -> float:
    """
    Deterministic hash function for user assignment.
    Returns a float in [0,1) for randomization.
    """
    h = hashlib.sha256((str(user_id) + salt).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

def assign_users(
    users_df: pd.DataFrame,
    exp_id: str,
    variant_list: list,
    strata_cols: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Assign users to experiment variants with stratification.
    """
    np.random.seed(seed)
    df = users_df.copy()
    df['bucket_ts'] = pd.Timestamp.now()
    
    # Create strata column
    if strata_cols:
        df['strata'] = df[strata_cols].astype(str).agg('_'.join, axis=1)
    else:
        df['strata'] = "all"
    
    # Use a single, vectorized operation for deterministic assignment
    df['hash_value'] = df['user_id'].apply(lambda uid: hash_user(uid, exp_id))
    num_variants = len(variant_list)
    df['variant_idx'] = (df['hash_value'] * num_variants).astype(int)
    df['variant'] = df['variant_idx'].apply(lambda i: variant_list[i])
    
    # Add exp_id to the DataFrame before creating the subset
    df['exp_id'] = exp_id

    assignments_df = df[['user_id','variant','bucket_ts','strata','exp_id']].copy()
    return assignments_df

def check_balance(assignments_df: pd.DataFrame, strata_col='strata') -> pd.DataFrame:
    """
    Checks balance of assignment per strata.
    Returns the fraction of users per variant in each strata.
    """
    balance = assignments_df.groupby([strata_col, 'variant']).size().unstack(fill_value=0)
    
    # Handle case where a strata has no users
    total = balance.sum(axis=1)
    if (total == 0).any():
        balance_frac = balance.div(total.replace(0,1), axis=0)
    else:
        balance_frac = balance.div(total, axis=0)
    return balance_frac
    
if __name__ == "__main__":
    # Example usage
    users = pd.DataFrame({
        'user_id': [f'u{i}' for i in range(1,21)],
        'country': ['US','IN','US','IN']*5,
        'device': ['mobile','desktop']*10
    })
    assignments = assign_users(users, 'exp_checkout_01', ['control','treatment'], strata_cols=['country','device'])
    print(assignments)
    print(check_balance(assignments))