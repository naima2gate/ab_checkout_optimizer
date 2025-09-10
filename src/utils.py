# src/utils.py
import pandas as pd
import numpy as np

def set_seed(seed=42):
    np.random.seed(seed)

def safe_mean(x):
    return np.nan if len(x)==0 else np.mean(x)

def one_hot_encode(df: pd.DataFrame, cols):
    return pd.get_dummies(df, columns=cols, drop_first=False)

def aggregate_by_user(df: pd.DataFrame, value_col='gpv', group_col='user_id'):
    return df.groupby(group_col, as_index=False)[value_col].sum()

def merge_with_default(df_left, df_right, on, default=0.0):
    merged = df_left.merge(df_right, on=on, how='left').fillna(default)
    return merged
