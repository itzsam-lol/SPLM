"""
Divergence Signal Constructor
Combines PAI (leading physical signal) and AMI (target: revenue surprise) 
to calculate the primary Alpha factor.
Synthetic: False
"""
import pandas as pd
import logging

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
try:
    from data.india_universe import INDIA_RETAIL_UNIVERSE
except ImportError:
    INDIA_RETAIL_UNIVERSE = {}

logger = logging.getLogger(__name__)

class DivergenceSignal:
    def __init__(self, data_source='revenue_surprise'):
        self.data_source = data_source

    def generate_signal(self, pai_df: pd.DataFrame, ami_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge PAI and Target (revenue surprise usually) dataframes 
        and calculate divergence factor.
        """
        if pai_df.empty or ami_df.empty:
            logger.warning("One of the input dataframes is empty. Cannot compute divergence.")
            return pd.DataFrame()

        merged = pd.merge(pai_df, ami_df, on=['ticker', 'year', 'quarter'], how='inner')
        
        if merged.empty:
            logger.warning("Intersection of PAI and Target arrays is empty.")
            return merged
            
        if self.data_source == 'revenue_surprise':
            # Δ = PAI_zscore (leading, physical) - revenue_surprise_zscore (target reality)
            # This measures how physical signals diverge from trailing reported reality
            target_col = 'revenue_surprise_zscore'
        else:
            target_col = 'ami_zscore'
            
        merged['delta_signal_raw'] = merged['pai_zscore'] - merged[target_col]
        
        def get_sector(t):
            return INDIA_RETAIL_UNIVERSE.get(t, {}).get('sector', 'unknown')
        
        merged['sector'] = merged['ticker'].apply(get_sector)
        
        # Cross-sectional ranking within sector
        merged['sector_rank'] = merged.groupby(['year', 'quarter', 'sector'])['delta_signal_raw'].rank(pct=True)
        
        # Add forward_target for IC calculation
        merged = merged.sort_values(['ticker', 'year', 'quarter'])
        merged['forward_revenue_surprise'] = merged.groupby('ticker')[target_col].shift(-1)
        
        merged['delta_signal_smooth'] = merged['delta_signal_raw'] # No ewma since quarterly
            
        return merged

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    args = parser.parse_args()
    
    # generate fake df and show
    pai = pd.DataFrame([{'ticker': args.ticker, 'year': 2022, 'quarter': 1, 'pai_zscore': 1.5, 'sector': 'hypermarket'}])
    ami = pd.DataFrame([{'ticker': args.ticker, 'year': 2022, 'quarter': 1, 'revenue_surprise_zscore': 0.5}])
    d = DivergenceSignal()
    print(d.generate_signal(pai, ami))
