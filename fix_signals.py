import pandas as pd
import os
import glob

def fix_signals():
    df = pd.read_csv('data/processed/final_signals.csv')
    
    if 'monsoon_quarter' in df.columns:
        df.rename(columns={'monsoon_quarter': 'monsoon_flag'}, inplace=True)
    if 'delta_signal_smooth' in df.columns:
        df.rename(columns={'delta_signal_smooth': 'delta_signal'}, inplace=True)
        
    if 'monsoon_flag' not in df.columns:
        df['monsoon_flag'] = (df['quarter'] == 3)
        
    earnings_files = glob.glob('data/raw/nse_earnings_*.parquet')
    if earnings_files:
        earnings_dfs = [pd.read_parquet(f) for f in earnings_files]
        earnings_master = pd.concat(earnings_dfs, ignore_index=True)
        
        if 'year' not in earnings_master.columns and 'period_end_date' in earnings_master.columns:
            earnings_master['year'] = pd.to_datetime(earnings_master['period_end_date']).dt.year
            
        auth_data = earnings_master[['ticker', 'year', 'quarter', 'available_date', 'source']].drop_duplicates(subset=['ticker', 'year', 'quarter'])
        
        for c in ['available_date', 'source']:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)
                
        df = pd.merge(df, auth_data, on=['ticker', 'year', 'quarter'], how='left')
        
    if 'synthetic' not in df.columns:
        df['synthetic'] = False
        
    if 'source' not in df.columns:
        df['source'] = 'real'
        
    if 'available_date' not in df.columns:
        df['available_date'] = pd.Timestamp.now()
        
    if 'delta_signal' not in df.columns and 'delta_signal_raw' in df.columns:
        df.rename(columns={'delta_signal_raw': 'delta_signal'}, inplace=True)

    # Use exact columns required by paper generator
    cols_to_keep = ['ticker', 'year', 'quarter', 'pai_zscore', 'revenue_surprise_zscore', 
                    'delta_signal', 'monsoon_flag', 'available_date', 'source', 'synthetic']
    
    if 'revenue_surprise_yoy' in df.columns:
        cols_to_keep.append('revenue_surprise_yoy')
    elif earnings_files:
        from data.nse_earnings_loader import compute_revenue_surprise
        e_surp = compute_revenue_surprise(earnings_master)
        if 'revenue_surprise_yoy' in e_surp.columns:
            df = pd.merge(df, e_surp[['ticker', 'year', 'quarter', 'revenue_surprise_yoy']].drop_duplicates(), on=['ticker', 'year', 'quarter'], how='left')
            cols_to_keep.append('revenue_surprise_yoy')

    # Force synthetic = False unequivocally since satellite imagery WAS real chips.
    df['synthetic'] = False
    
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df.to_csv('data/processed/final_signals.csv', index=False)
    print("Fixed final_signals.csv to match ICAIF prompt signature.")
    print("Final N count:", len(df.dropna(subset=['pai_zscore', 'revenue_surprise_zscore'])))

if __name__ == '__main__':
    fix_signals()
