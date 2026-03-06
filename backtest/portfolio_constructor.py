"""
Portfolio Constructor.
Uses CVXPY to build a sector-neutral Long/Short portfolio in INR.
"""
import pandas as pd
import numpy as np
import logging
import pandas_market_calendars as mcal

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class PortfolioConstructor:
    def __init__(self, target_gross_exposure: float = 2.0, max_sector_net: float = 0.30, max_position: float = 0.03):
        """
        target_gross_exposure: 2.0
        max_sector_net: 30% sector cap
        max_position: 3% single ticker
        """
        self.target_gross = target_gross_exposure
        self.max_sector_net = max_sector_net
        self.max_pos = max_position
        # Handle lack of market calendar gracefully in tests
        try:
            self.calendar = mcal.get_calendar('NSE')
        except:
            self.calendar = None

    def filter_liquidity(self, df: pd.DataFrame, daily_volume_inr_cr: pd.Series) -> pd.DataFrame:
        """
        Remove tickers where avg daily volume < ₹5 crore
        """
        valid_tickers = daily_volume_inr_cr[daily_volume_inr_cr >= 5.0].index
        return df[df['ticker'].isin(valid_tickers)]

    def get_rebalance_date(self, earnings_announce_date: pd.Timestamp) -> pd.Timestamp:
        """
        Rebalance is quarterly, 5 trading days after earnings for a given name/cohort.
        """
        if pd.isna(earnings_announce_date): return pd.NaT
        end_dt = earnings_announce_date + pd.Timedelta(days=15)
        
        if self.calendar:
            valid_days = self.calendar.valid_days(start_date=earnings_announce_date, end_date=end_dt)
            if len(valid_days) > 5:
                return valid_days[5]
        return earnings_announce_date + pd.Timedelta(days=5)

    def construct_portfolio(self, signal_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
        df = signal_df.copy()
        n_assets = len(df)
        if n_assets < 5:
            return pd.DataFrame()

        alphas = df['delta_signal_smooth'].values
        industries = df['sector'].unique() if 'sector' in df.columns else ['unknown']
        sector_matrix = np.zeros((len(industries), n_assets))
        if 'sector' in df.columns:
            for i, ind in enumerate(industries):
                sector_matrix[i, :] = (df['sector'] == ind).astype(float)

        if CVXPY_AVAILABLE:
            weights = cp.Variable(n_assets)
            objective = cp.Maximize(alphas @ weights)
            constraints = [
                cp.sum(weights) == 0, # Neutral
                cp.norm(weights, 1) <= self.target_gross,
                weights <= self.max_pos,
                weights >= -self.max_pos
            ]
            for i in range(len(industries)):
                sector_exposure = sector_matrix[i, :] @ weights
                constraints.append(sector_exposure <= self.max_sector_net)
                constraints.append(sector_exposure >= -self.max_sector_net)
                
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.ECOS)
                if weights.value is not None:
                    df['weight'] = weights.value
                else:
                    df['weight'] = self._fallback_weights(df)
            except:
                df['weight'] = self._fallback_weights(df)
        else:
            df['weight'] = self._fallback_weights(df)
            
        df['weight'] = df['weight'].apply(lambda x: 0 if abs(x) < 1e-4 else x)
        df['rebalance_date'] = rebalance_date
        return df[['rebalance_date', 'ticker', 'weight', 'delta_signal_smooth']]

    def _fallback_weights(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        alphas = df['delta_signal_smooth'].values
        top_quintile_thresh = np.percentile(alphas, 80)
        bot_quintile_thresh = np.percentile(alphas, 20)
        
        weights = np.zeros(n)
        long_idx = alphas >= top_quintile_thresh
        short_idx = alphas <= bot_quintile_thresh
        
        n_long = np.sum(long_idx)
        n_short = np.sum(short_idx)
        leg_exp = self.target_gross / 2.0
        
        if n_long > 0: weights[long_idx] = leg_exp / n_long
        if n_short > 0: weights[short_idx] = -leg_exp / n_short
            
        return np.clip(weights, -self.max_pos, self.max_pos)
