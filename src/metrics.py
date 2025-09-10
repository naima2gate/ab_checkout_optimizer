# src/metrics.py
import pandas as pd
import numpy as np

def compute_gpv(orders: pd.DataFrame) -> pd.Series:
    """
    Compute Gross Profit per Order (GPV).
    """
    orders = orders.copy()
    orders['gpv'] = orders['revenue'] - orders['discount'] - orders['var_cost']
    return orders.set_index('session_id')['gpv']

def conversion_rate(events: pd.DataFrame, event_name='place_order') -> pd.DataFrame:
    """
    Compute conversion rate per user.
    """
    conv = events.groupby('user_id')['name'].apply(lambda x: int(event_name in x.values))
    return conv.reset_index(name='conversion')

def guardrails(sessions: pd.DataFrame, orders: pd.DataFrame, events: pd.DataFrame, perf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute guardrail metrics per variant.
    """
    # Get all unique variants
    all_variants = sessions['variant'].drop_duplicates()
    
    # Checkout latency
    latency = perf.merge(sessions[['session_id','variant']], on='session_id', how='left')
    latency_summary = latency.groupby('variant')['checkout_latency_ms'].quantile(0.95).reset_index()
    
    # Refund rate
    orders_count = orders.merge(sessions[['session_id','variant']], on='session_id', how='left').groupby('variant').size()
    refunds_count = events[events['name']=='refund'].merge(sessions[['session_id','variant']], on='session_id', how='left').groupby('variant').size()
    
    # Use reindex to ensure all variants are present with 0 counts if no refunds
    refund_rate = (refunds_count.reindex(all_variants, fill_value=0) / orders_count.reindex(all_variants, fill_value=0) * 100).reset_index(name='value')
    
    # Support tickets per 1k orders
    tickets_count = events[events['name']=='support_ticket'].merge(sessions[['session_id','variant']], on='session_id', how='left').groupby('variant').size()
    support_rate = (tickets_count.reindex(all_variants, fill_value=0) / orders_count.reindex(all_variants, fill_value=0) * 1000).reset_index(name='value')
    
    # Build the final DataFrame
    latency_summary['metric'] = 'checkout_latency_p95'
    refund_rate['metric'] = 'refund_rate'
    support_rate['metric'] = 'support_tickets_per_1k_orders'
    
    guardrail_df = pd.concat([
        latency_summary.rename(columns={'checkout_latency_ms':'value'}),
        refund_rate,
        support_rate
    ], ignore_index=True)
    
    return guardrail_df

def summarize_metrics(sessions, orders, events, perf):
    """
    Returns a dictionary with GPV, conversion, and guardrails per variant.
    """
    # GPV per user
    gpv = compute_gpv(orders).reset_index()
    gpv = gpv.merge(sessions[['user_id','variant']].drop_duplicates(), on='user_id', how='left')
    gpv_summary = gpv.groupby('variant')['gpv'].sum().reset_index()
    
    # Conversion
    conv = conversion_rate(events)
    conv = conv.merge(sessions[['user_id','variant']].drop_duplicates(), on='user_id', how='left')
    conv_summary = conv.groupby('variant')['conversion'].mean().reset_index()
    
    # Guardrails
    guard = guardrails(sessions, orders, events, perf)
    
    return {
        'gpv': gpv_summary,
        'conversion': conv_summary,
        'guardrails': guard
    }

if __name__ == "__main__":
    # Example usage
    sessions = pd.DataFrame({
        'session_id': [f's{i}' for i in range(1,6)],
        'user_id': [f'u{i}' for i in range(1,6)],
        'variant': ['control','treatment','control','treatment','control']
    })
    orders = pd.DataFrame({
        'order_id': [1,2,3,4,5],
        'session_id': [f's{i}' for i in range(1,6)],
        'revenue': [100,120,90,110,95],
        'discount': [10,5,0,15,5],
        'var_cost': [20,25,15,20,15]
    })
    events = pd.DataFrame({
        'session_id': [f's{i}' for i in range(1,6)],
        'user_id': [f'u{i}' for i in range(1,6)],
        'name': ['place_order','place_order','refund','support_ticket','place_order']
    })
    perf = pd.DataFrame({
        'session_id': [f's{i}' for i in range(1,6)],
        'checkout_latency_ms': [200,300,250,400,150]
    })
    
    summary = summarize_metrics(sessions, orders, events, perf)
    print(summary)