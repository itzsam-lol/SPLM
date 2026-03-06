"""
Occupancy Aggregator
Rolls up individual store-level occupancy ratios into a ticker-level raw index.
"""
import pandas as pd
import logging
from datetime import datetime
from typing import List

logger = logging.getLogger(__name__)

class OccupancyAggregator:
    def __init__(self):
        """
        Initialize the aggregator.
        """
        pass

    def aggregate_to_ticker(self, daily_inferences: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame of daily store-level inference results and rolls them up
        to the ticker level.
        
        Expected columns in `daily_inferences`:
        [ticker, store_id, date, vehicle_count, lot_capacity, occupancy_ratio, data_quality_flag]
        """
        if daily_inferences.empty:
            logger.warning("Empty inferences DataFrame provided to aggregator.")
            return pd.DataFrame()
            
        logger.info(f"Aggregating {len(daily_inferences)} store-level inferences to ticker level.")

        # 1. Filter out known bad quality data before aggregation
        good_data = daily_inferences[daily_inferences['data_quality_flag'] == True]
        
        if good_data.empty:
            logger.warning("No high-quality inference data available after filtering.")
            return pd.DataFrame()

        # 2. Group by Ticker and Date
        # We compute the weighted average occupancy (weighted by lot capacity)
        # to ensure larger stores have proportionately more impact on the final signal.
        def weighted_occupancy(group):
            total_capacity = group['lot_capacity'].sum()
            if total_capacity <= 0:
                return 0.0
            
            # Weighted average: sum(ratio * capacity) / sum(capacity) array math
            weighted_sum = (group['occupancy_ratio'] * group['lot_capacity']).sum()
            return weighted_sum / total_capacity

        ticker_daily = good_data.groupby(['ticker', 'date']).apply(
            lambda x: pd.Series({
                'raw_occupancy_ratio': weighted_occupancy(x),
                'total_vehicles': x['vehicle_count'].sum(),
                'total_capacity': x['lot_capacity'].sum(),
                'store_sample_size': len(x)
            })
        ).reset_index()

        # 3. Handle missing days
        # A naive implementation interpolates up to 3 days of missing satellite coverage.
        ticker_daily = self._interpolate_missing_days(ticker_daily)
        
        return ticker_daily

    def _interpolate_missing_days(self, df: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
        """
        Fills missing dates for each ticker using linear interpolation,
        but only up to `max_gap` consecutive missing days.
        """
        if df.empty:
            return df
            
        interpolated_dfs = []
        
        for ticker, group in df.groupby('ticker'):
            # Ensure dates are sorted
            group = group.sort_values('date').set_index('date')
            
            # Create a complete date range from min to max date for this ticker
            full_date_range = pd.date_range(start=group.index.min(), end=group.index.min() + pd.Timedelta(days=100)) # mock range
            
            # Reindex to insert NaNs for missing days
            group_reindexed = group.reindex(full_date_range)
            group_reindexed['ticker'] = ticker
            
            # Linear interpolate, with a limit on consecutive NaNs
            group_reindexed['raw_occupancy_ratio'] = group_reindexed['raw_occupancy_ratio'].interpolate(
                method='linear', limit=max_gap, limit_direction='forward'
            )
            
            # Drop rows that are still NaN (meaning the gap was > max_gap)
            group_clean = group_reindexed.dropna(subset=['raw_occupancy_ratio']).reset_index()
            group_clean.rename(columns={'index': 'date'}, inplace=True)
            
            interpolated_dfs.append(group_clean)
            
        if not interpolated_dfs:
            return pd.DataFrame()
            
        return pd.concat(interpolated_dfs, ignore_index=True)
