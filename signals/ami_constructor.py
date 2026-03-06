"""
Analyst Momentum Index (AMI) Constructor.
Extracts signal from the velocity and direction of analyst estimate revisions.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AMIConstructor:
    def __init__(self, ibes_df: pd.DataFrame):
        """
        Initialize the constructor with a pre-loaded dataframe of strictly point-in-time
        revisions.
        """
        self.ibes_df = ibes_df

    def compute_broker_weight(self, aum_coverage_rank: int) -> float:
        """
        Weight broker influence using a log transformation of their rank.
        Real implementations would use external data for broker AUM.
        Here we mock an inverse log relation where rank 1 has highest weight.
        """
        if aum_coverage_rank <= 0:
            return 1.0
        return max(0.1, np.log1p(100 / aum_coverage_rank) / 5.0)

    def compute_ami(self, ticker: str, as_of_date: datetime, lookback_days: int = 20) -> dict:
        """
        Compute the Analyst Momentum Index for a given ticker on a specific date.
        """
        start_date = as_of_date - pd.Timedelta(days=lookback_days)
        
        # 1. Isolate the lookback window
        window = self.ibes_df[
            (self.ibes_df['ticker'] == ticker) &
            (self.ibes_df['revision_date'] >= start_date) &
            (self.ibes_df['revision_date'] <= as_of_date)
        ]
        
        if window.empty:
            return {
                "ami_raw": 0.0,
                "ami_period_type": "inter_quarter",
                "revisions_count": 0
            }
            
        # Add weights if not present (mocking a rank if needed)
        if 'broker_weight' not in window.columns:
            window['broker_weight'] = 1.0
            
        # 2. Compute weighted upgrades and downgrades
        upgrades = window[window['direction'] == 'up']['broker_weight'].sum()
        downgrades = window[window['direction'] == 'down']['broker_weight'].sum()
        
        # 3. Compute ratio
        # Bounded between -1.0 and 1.0
        total_revisions = upgrades + downgrades
        
        if total_revisions == 0:
             raw_ami = 0.0
        else:
             raw_ami = (upgrades - downgrades) / total_revisions
             
        # Mocking earnings proximity logic (simplification)
        # Assuming if revisions clustered within 10 days of a month start it's "pre-earnings"
        is_pre_earnings = "pre_earnings" if as_of_date.day <= 10 else "inter_quarter"
        
        return {
            "ami_raw": float(raw_ami),
            "ami_period_type": is_pre_earnings,
            "revisions_count": int(len(window))
        }

    def build_ami_panel(self, target_dates: list) -> pd.DataFrame:
        """
        Build the full AMI dataframe for a range of dates.
        """
        tickers = self.ibes_df['ticker'].unique()
        records = []
        
        for date in target_dates:
            for ticker in tickers:
                metrics = self.compute_ami(ticker, date)
                records.append({
                    "ticker": ticker,
                    "date": date,
                    "ami_raw": metrics['ami_raw'],
                    "ami_period_type": metrics['ami_period_type'],
                    "revisions_count": metrics['revisions_count']
                })
                
        df = pd.DataFrame(records)
        
        # 4. Cross-sectional Z-Score calculation (Winsorized)
        # To prevent extreme outliers from single broker drops
        def winsorize_zscore(group):
            s = group['ami_raw']
            lower = s.quantile(0.05)
            upper = s.quantile(0.95)
            s_clipped = s.clip(lower, upper)
            
            if s_clipped.std() == 0:
                return pd.Series(0, index=s.index)
                
            return (s_clipped - s_clipped.mean()) / s_clipped.std()
            
        df['ami_zscore'] = df.groupby('date').apply(winsorize_zscore).reset_index(level=0, drop=True)
        return df
