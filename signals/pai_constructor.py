"""
Physical Activity Index (PAI) Constructor.
Takes raw aggregated occupancy ratios and normalizes them against weather and calendar effects.
Updated for Indian quarterly retail data.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PAIConstructor:
    def __init__(self):
        pass

    def load_weather_data(self, tickers: list, years: list, quarters: list) -> pd.DataFrame:
        """
        Mock: Load external weather data for a given quarter.
        """
        records = []
        for ticker in tickers:
            for y in years:
                for q in quarters:
                    records.append({
                        "ticker": ticker,
                        "year": y,
                        "quarter": q,
                        "precipitation_mm": np.random.exponential(scale=2.0),
                        "temp_deviation": np.random.normal(loc=0.0, scale=5.0),
                        "is_diwali_quarter": 1 if q == 4 else 0, # Usually Q4 (Oct-Dec)
                        "is_navratri": 1 if q in [3, 4] else 0,
                        "is_eid": 1 if q in [1, 2, 3] else 0 # Mock generic variability
                    })
        return pd.DataFrame(records)

    def _apply_weather_normalization(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits an OLS regression per ticker to residualize occupancy against external confounders.
        """
        panel_df['log_occupancy'] = np.log(panel_df['raw_occupancy_ratio'] + 1e-6)
        normalized_dfs = []
        
        for ticker, group in panel_df.groupby('ticker'):
            if len(group) < 8:
                logger.warning(f"Not enough data for {ticker} weather normalization.")
            try:
                model = smf.ols(
                    'log_occupancy ~ precipitation_mm + temp_deviation + '
                    'is_diwali_quarter + is_navratri + is_eid',
                    data=group
                ).fit()
                group['pai_raw'] = model.resid
                normalized_dfs.append(group)
            except Exception as e:
                logger.error(f"OLS failure for {ticker}: {e}")
                group['pai_raw'] = np.nan
                normalized_dfs.append(group)
                
        if not normalized_dfs:
            return panel_df
        return pd.concat(normalized_dfs, ignore_index=True)

    def build_pai(self, aggregated_occupancy: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline to build the Physical Activity Index.
        aggregated_occupancy must have: [ticker, year, quarter, raw_occupancy_ratio]
        """
        if aggregated_occupancy.empty:
            return pd.DataFrame()
            
        tickers = aggregated_occupancy['ticker'].unique()
        years = aggregated_occupancy['year'].unique()
        quarters = aggregated_occupancy['quarter'].unique()
        weather_df = self.load_weather_data(tickers, years, quarters)
        
        panel_df = pd.merge(aggregated_occupancy, weather_df, on=['ticker', 'year', 'quarter'], how='inner')
        panel_df = self._apply_weather_normalization(panel_df)
        
        # 4 quarters lookback rolling zscore
        # Compute zscore of pai_raw relative to the past 4 quarters
        def rolling_zscore(x, window=4):
            r = x.rolling(window=window, min_periods=2)
            return (x - r.mean()) / (r.std() + 1e-8)
            
        panel_df = panel_df.sort_values(['ticker', 'year', 'quarter'])
        
        panel_df['monsoon_quarter'] = panel_df['quarter'].apply(lambda q: q == 3)
        
        # We compute z-score on all data
        panel_df['pai_zscore_all'] = (
            panel_df.groupby('ticker')['pai_raw']
            .transform(lambda x: rolling_zscore(x))
        )
        
        # Nullify monsoon quarters
        panel_df['pai_zscore'] = panel_df.apply(
            lambda row: np.nan if row['monsoon_quarter'] else row['pai_zscore_all'], axis=1
        )
        
        output_cols = ['ticker', 'year', 'quarter', 'pai_raw', 'pai_zscore', 'monsoon_quarter']
        return panel_df[output_cols].dropna(subset=['pai_raw'])
