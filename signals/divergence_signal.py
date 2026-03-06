"""
Divergence Signal Constructor
Combines PAI and AMI to calculate the primary Alpha factor.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DivergenceSignal:
    def __init__(self, use_smoothing: bool = True):
        """
        Initialize setup.
        """
        self.use_smoothing = use_smoothing

    def generate_signal(self, pai_df: pd.DataFrame, ami_df: pd.DataFrame, industry_map: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge PAI and AMI dataframes and calculate the divergence factor.
        
        industry_map: DataFrame with [ticker, gics_sub_ind] to allow within-sector ranking.
        """
        logger.info("Generating PAI-AMI Divergence Signal")
        
        # 1. Merge the two signals
        merged = pd.merge(pai_df, ami_df, on=['ticker', 'date'], how='inner')
        
        if merged.empty:
            logger.warning("Intersection of PAI and AMI arrays is empty.")
            return merged
            
        # 2. Raw Divergence Calculation
        # Positive value means physical activity (PAI) is stronger than 
        # what analysts are modeling (AMI)
        merged['delta_signal_raw'] = merged['pai_zscore'] - merged['ami_zscore']
        
        # 3. Add Industry Data (Mock mapping here if None)
        if industry_map is None:
            # Mock generic sector
            merged['industry'] = [(hash(t) % 5) for t in merged['ticker']]
        else:
            merged = pd.merge(merged, industry_map, on='ticker', how='left')
        
        # 4. Cross-sectional ranking within industry
        # Rank from 0 to 1, where 1 is the most long-biased
        merged['industry_rank'] = merged.groupby(['date', 'industry'])['delta_signal_raw'].rank(pct=True)
        
        # 5. Smoothing
        # Apply 5-day Exponentially Weighted Moving Average (EWMA) with halflife=3
        # to smooth out noise from day-to-day satellite fluctuations.
        if self.use_smoothing:
            merged = merged.sort_values(['ticker', 'date'])
            merged['delta_signal_smooth'] = (
                merged.groupby('ticker')['delta_signal_raw']
                .transform(lambda x: x.ewm(halflife=3, min_periods=1).mean())
            )
        else:
            merged['delta_signal_smooth'] = merged['delta_signal_raw']
            
        return merged
