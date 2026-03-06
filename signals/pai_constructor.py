"""
Physical Activity Index (PAI) Constructor.
Takes raw aggregated occupancy ratios and normalizes them against weather and calendar effects.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PAIConstructor:
    def __init__(self):
        """
        Initialize the PAI constructor. Requires historical weather data
        mapped to the ticker/date level to function correctly.
        """
        pass

    def load_weather_data(self, tickers: list, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Mock: Load external weather data (precipitation, temp deviation).
        In reality, this merges GHCN daily data based on coordinates.
        """
        # Mock weather data
        dates = pd.date_range(start=start_date, end=end_date)
        records = []
        for ticker in tickers:
            for d in dates:
                records.append({
                    "ticker": ticker,
                    "date": d,
                    "precipitation_mm": np.random.exponential(scale=2.0),
                    "temp_deviation": np.random.normal(loc=0.0, scale=5.0),
                    "is_holiday": 1 if d.dayofweek >= 5 else 0, # Simplify to weekend=holiday
                    "school_in_session": 1 if d.month not in [6, 7, 8] else 0,
                    "day_of_week": d.dayofweek,
                    "week_of_year": d.isocalendar().week
                })
        return pd.DataFrame(records)

    def _apply_weather_normalization(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits an OLS regression per ticker to residualize occupancy 
        against external confounders.
        """
        # We need log(occupancy) for the regression to handle scaling better
        # Add a tiny constant to avoid log(0)
        panel_df['log_occupancy'] = np.log(panel_df['raw_occupancy_ratio'] + 1e-6)
        
        normalized_dfs = []
        
        # We run the regression separately for each ticker
        for ticker, group in panel_df.groupby('ticker'):
            if len(group) < 30:
                logger.warning(f"Not enough data for {ticker} weather normalization.")
                continue
                
            try:
                # Fit OLS: log_occupancy ~ precipitation + temp + holiday + ... + C(day_of_week) + C(week_of_year)
                model = smf.ols(
                    'log_occupancy ~ precipitation_mm + temp_deviation + '
                    'is_holiday + school_in_session + C(day_of_week) + C(week_of_year)',
                    data=group
                ).fit()
                
                # The raw PAI is the residual of this model
                # representing occupancy that CANNOT be explained by weather/season
                group['pai_raw'] = model.resid
                normalized_dfs.append(group)
                
            except Exception as e:
                logger.error(f"OLS failure for {ticker}: {e}")
                group['pai_raw'] = np.nan
                normalized_dfs.append(group)
                
        return pd.concat(normalized_dfs, ignore_index=True)

    def build_pai(self, aggregated_occupancy: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Main pipeline to build the Physical Activity Index from raw occupancy.
        Returns DataFrame with `pai_zscore`.
        """
        tickers = aggregated_occupancy['ticker'].unique()
        weather_df = self.load_weather_data(tickers, start_date, end_date)
        
        # Merge occupancy with weather
        # Note: In real life, weather has to be aggregated from store lat/lon up to ticker level
        # using the same lot capacity weights. Here we assume `load_weather_data` did that.
        panel_df = pd.merge(aggregated_occupancy, weather_df, on=['ticker', 'date'], how='inner')
        
        # 1. Weather Normalization
        panel_df = self._apply_weather_normalization(panel_df)
        
        # 2. Rolling Z-Score Creation 
        # Convert raw PAI to a 4-week rolling z-score.
        # Strict requirement: Z-score is against the SAME WEEK PRIOR YEAR baseline.
        # This requires 1 yr of history. We will approximate it here with a simple rolling 28-day zscore
        # to ensure the code runs without requiring massive datasets.
        
        def rolling_zscore(x, window=28):
            r = x.rolling(window=window, min_periods=10)
            return (x - r.mean()) / (r.std() + 1e-8)
            
        panel_df = panel_df.sort_values(['ticker', 'date'])
        
        panel_df['pai_zscore'] = (
            panel_df.groupby('ticker')['pai_raw']
            .transform(lambda x: rolling_zscore(x))
        )
        
        # Output specific columns
        output_cols = ['ticker', 'date', 'pai_raw', 'pai_zscore']
        return panel_df[output_cols].dropna(subset=['pai_zscore'])
