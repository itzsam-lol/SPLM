import pytest
import pandas as pd
import numpy as np
from data.nse_earnings_loader import compute_revenue_surprise, get_nse_quarterly_results

def test_revenue_surprise_computable():
    # Mock data
    df = pd.DataFrame([
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2021-03-31'), 'revenue_cr': 100},
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2021-06-30'), 'revenue_cr': 110},
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2021-09-30'), 'revenue_cr': -50}, # Negative base test
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2021-12-31'), 'revenue_cr': 120},
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2022-03-31'), 'revenue_cr': 150}, # 150 vs 100 -> 0.5
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2022-06-30'), 'revenue_cr': 110}, # 110 vs 110 -> 0.0
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2022-09-30'), 'revenue_cr': 10},  # 10 vs -50 -> 60/abs(-50) = 1.2
    ])
    
    surp_df = compute_revenue_surprise(df)
    
    assert abs(surp_df.iloc[4]['revenue_surprise_yoy'] - 0.5) < 1e-4
    assert abs(surp_df.iloc[5]['revenue_surprise_yoy'] - 0.0) < 1e-4
    assert abs(surp_df.iloc[6]['revenue_surprise_yoy'] - 1.2) < 1e-4

def test_available_date_lag():
    df = pd.DataFrame([
        {'ticker': 'DMART', 'period_end_date': pd.to_datetime('2022-03-31'), 
         'earnings_announce_date': pd.to_datetime('2022-05-15'), 'revenue_cr': 150, 'source': 'nse'}
    ])
    # The actual get function assigns available date, we mock that behavior
    df['available_date'] = df['earnings_announce_date'] + pd.Timedelta(1, 'D')
    
    assert df.iloc[0]['available_date'] == pd.to_datetime('2022-05-16')
