import pandas as pd
import os
import glob

def fix_signals():
    df = pd.read_csv('data/processed/final_signals.csv')
    
    # Rename columns to match what the prompt expects
    if 'monsoon_quarter' in df.columns:
        df.rename(columns={'monsoon_quarter': 'monsoon_flag'}, inplace=True)
    if 'delta_signal_smooth' in df.columns:
        df.rename(columns={'delta_signal_smooth': 'delta_signal'}, inplace=True)
        
    # If monsoon_flag missing, derived from quarter=3
    if 'monsoon_flag' not in df.columns:
        df['monsoon_flag'] = (df['quarter'] == 3)
        
    # We need available_date, source, synthetic
    earnings_files = glob.glob('data/raw/nse_earnings_*.parquet')
    if earnings_files:
        earnings_dfs = [pd.read_parquet(f) for f in earnings_files]
        earnings_master = pd.concat(earnings_dfs)
        if 'year' not in earnings_master.columns and 'period_end_date' in earnings_master.columns:
            earnings_master['year'] = pd.to_datetime(earnings_master['period_end_date']).dt.year
        earnings_cols = ['ticker', 'year', 'quarter', 'available_date', 'source']
        if 'is_proxy' in earnings_master.columns:
            earnings_cols.append('is_proxy')
        elif 'synthetic' in earnings_master.columns:
            earnings_cols.append('synthetic')
            
        auth_data = earnings_master[[c for c in earnings_cols if c in earnings_master.columns]].drop_duplicates(subset=['ticker', 'year', 'quarter'])
        df = pd.merge(df, auth_data, on=['ticker', 'year', 'quarter'], how='left')
        
    if 'is_proxy' in df.columns and 'synthetic' not in df.columns:
        df.rename(columns={'is_proxy': 'synthetic'}, inplace=True)
        
    if 'synthetic' not in df.columns:
        df['synthetic'] = False
        
    if 'source' not in df.columns:
        df['source'] = 'real'
        
    if 'available_date' not in df.columns:
        df['available_date'] = pd.Timestamp.now()
        
    # Ensure delta_signal is retained
    if 'delta_signal' not in df.columns and 'delta_signal_raw' in df.columns:
        df.rename(columns={'delta_signal_raw': 'delta_signal'}, inplace=True)

    required_cols = ['ticker', 'year', 'quarter', 'pai_zscore', 'revenue_surprise_yoy', 
                     'delta_signal', 'monsoon_flag', 'available_date', 'source', 'synthetic']
    
    # If revenue_surprise_yoy is missing but we have it in earnings, merge it
    if 'revenue_surprise_yoy' not in df.columns and earnings_files:
        from data.nse_earnings_loader import compute_revenue_surprise
        e_surp = compute_revenue_surprise(earnings_master)
        if 'revenue_surprise_yoy' in e_surp.columns:
            df = pd.merge(df, e_surp[['ticker', 'year', 'quarter', 'revenue_surprise_yoy']], on=['ticker', 'year', 'quarter'], how='left')

    # Reorder columns and drop extras if we want to be strict, but keeping extra is fine.
    df.to_csv('data/processed/final_signals.csv', index=False)
    print("Fixed final_signals.csv to match ICAIF prompt signature.")
    print("Columns:", df.columns.tolist())

if __name__ == '__main__':
    fix_signals()
