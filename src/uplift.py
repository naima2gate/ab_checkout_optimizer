# src/uplift.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def t_learner(df: pd.DataFrame, outcome_col: str, treat_col: str, feature_cols: list, random_state=42):
    """
    T-Learner: separate models for control and treatment to estimate CATE
    """
    df_train, df_pred = train_test_split(df, test_size=0.3, random_state=random_state)
    
    # Split control and treatment
    control_df = df_train[df_train[treat_col]==0].copy()
    treatment_df = df_train[df_train[treat_col]==1].copy()
    
    # Train models
    model_c = GradientBoostingRegressor(random_state=random_state)
    model_c.fit(control_df[feature_cols], control_df[outcome_col])
    
    model_t = GradientBoostingRegressor(random_state=random_state)
    model_t.fit(treatment_df[feature_cols], treatment_df[outcome_col])
    
    # Predict on the full prediction set
    y0_pred = model_c.predict(df_pred[feature_cols])
    y1_pred = model_t.predict(df_pred[feature_cols])
    
    cate = y1_pred - y0_pred
    df_pred = df_pred.copy()
    df_pred['cate'] = cate
    return df_pred[['user_id','cate']]

def x_learner(df: pd.DataFrame, outcome_col: str, treat_col: str, feature_cols: list, random_state=42):
    """
    X-Learner: estimate heterogeneous treatment effects (CATE).
    
    This function implements the standard X-Learner approach:
    1. Model the outcome for the control and treatment groups separately.
    2. Estimate the conditional average treatment effect (CATE) for each user.
    3. Model the propensity score for each user.
    4. Use the propensity score to combine the two CATE models into a final, weighted estimate.
    """
    df_train, df_pred = train_test_split(df, test_size=0.3, random_state=random_state)

    # 1. Model the outcome for control and treatment groups
    control_df = df_train[df_train[treat_col]==0].copy()
    treatment_df = df_train[df_train[treat_col]==1].copy()
    
    model_c = GradientBoostingRegressor(random_state=random_state)
    model_c.fit(control_df[feature_cols], control_df[outcome_col])
    
    model_t = GradientBoostingRegressor(random_state=random_state)
    model_t.fit(treatment_df[feature_cols], treatment_df[outcome_col])

    # 2. Estimate CATE for each user
    tau_t = df_train[outcome_col] - model_c.predict(df_train[feature_cols])
    tau_c = model_t.predict(df_train[feature_cols]) - df_train[outcome_col]

    # 3. Model the propensity score
    propensity_model = GradientBoostingClassifier(random_state=random_state)
    propensity_model.fit(df_train[feature_cols], df_train[treat_col])
    propensity_scores = propensity_model.predict_proba(df_pred[feature_cols])[:, 1]
    
    # 4. Use propensity scores to combine the models
    cate = propensity_scores * model_t.predict(df_pred[feature_cols]) - (1 - propensity_scores) * model_c.predict(df_pred[feature_cols])
    
    df_pred = df_pred.copy()
    df_pred['cate'] = cate
    return df_pred[['user_id', 'cate']]

def uplift_summary(cate_df: pd.DataFrame, threshold=0.0):
    """
    Summarize uplift: proportion of users with positive lift and top segments
    """
    total = len(cate_df)
    positive = (cate_df['cate'] > threshold).sum()
    top_users = cate_df.sort_values('cate', ascending=False).head(int(0.1*total))
    
    return {
        'total_users': total,
        'positive_lift_pct': float(positive/total),
        'top_10pct_users_summary': top_users.describe()
    }
    
if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'user_id': [f'u{i}' for i in range(n)],
        'variant': np.random.choice([0,1], n),
        'feature1': np.random.normal(50, 10, n),
        'feature2': np.random.normal(100, 20, n)
    })
    
    # Create a true CATE signal
    df['outcome'] = 0
    df.loc[df['variant']==1, 'outcome'] = 5 + 0.5 * (df.loc[df['variant']==1, 'feature1'] - 50) + np.random.normal(0,1,len(df.loc[df['variant']==1]))
    df.loc[df['variant']==0, 'outcome'] = np.random.normal(0,1,len(df.loc[df['variant']==0]))

    t_res = t_learner(df, 'outcome', 'variant', ['feature1','feature2'])
    x_res = x_learner(df, 'outcome', 'variant', ['feature1','feature2'])
    
    print("T-Learner Summary:")
    print(uplift_summary(t_res))
    print("X-Learner Summary:")
    print(uplift_summary(x_res))