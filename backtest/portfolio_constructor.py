"""
Portfolio Constructor.
Uses CVXPY to build a sector-neutral Long/Short portfolio from the Alpha signal.
"""
import pandas as pd
import numpy as np
import logging

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class PortfolioConstructor:
    def __init__(self, target_gross_exposure: float = 2.0, max_sector_net: float = 0.05, max_position: float = 0.03):
        """
        Initialize constraints.
        target_gross_exposure: 2.0 means 100% long, 100% short.
        max_sector_net: Sector neutrality constraint (e.g. +/- 5% per sector)
        max_position: Maximum weight in a single ticker
        """
        self.target_gross = target_gross_exposure
        self.max_sector_net = max_sector_net
        self.max_pos = max_position

    def construct_portfolio(self, signal_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
        """
        Given a dataframe of signals for a specific rebalance date, compute
        optimal portfolio weights.
        
        signal_df must have: ['ticker', 'delta_signal_smooth', 'industry']
        """
        # Filter to target date and ensure no missing signals
        df = signal_df[signal_df['date'] == rebalance_date].copy()
        df = df.dropna(subset=['delta_signal_smooth', 'industry'])
        
        n_assets = len(df)
        if n_assets < 20: # Arbitrary minimum to form a balanced L/S
            logger.warning(f"Insufficient assets ({n_assets}) to form portfolio on {rebalance_date}")
            return pd.DataFrame()

        # Simplify problem: We want to maximize signal exposure subject to constraints.
        alphas = df['delta_signal_smooth'].values
        
        # Sector/Industry dummy matrix
        industries = df['industry'].unique()
        sector_matrix = np.zeros((len(industries), n_assets))
        for i, ind in enumerate(industries):
            sector_matrix[i, :] = (df['industry'] == ind).astype(float)

        if CVXPY_AVAILABLE:
            # 1. Variables
            weights = cp.Variable(n_assets)
            
            # 2. Objective: Maximize alpha
            objective = cp.Maximize(alphas @ weights)
            
            # 3. Constraints
            constraints = [
                cp.sum(weights) == 0, # Dollar neutral
                cp.norm(weights, 1) <= self.target_gross, # Gross exposure limit
                weights <= self.max_pos, # Max long position
                weights >= -self.max_pos # Max short position
            ]
            
            # Sector neutrality constraints
            for i in range(len(industries)):
                sector_exposure = sector_matrix[i, :] @ weights
                constraints.append(sector_exposure <= self.max_sector_net)
                constraints.append(sector_exposure >= -self.max_sector_net)
                
            # 4. Solve Formulation
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve(solver=cp.ECOS)
                if weights.value is None:
                    raise Exception("Optimization failed to find a valid solution.")
                    
                df['weight'] = weights.value
                
            except Exception as e:
                logger.error(f"CVXPY Optimization failed on {rebalance_date}: {e}")
                df['weight'] = self._fallback_heuristic_weights(df)
                
        else:
            logger.info("CVXPY not available. Using heuristic rank-based weighting.")
            df['weight'] = self._fallback_heuristic_weights(df)
            
        # Clean up very small weights
        df['weight'] = df['weight'].apply(lambda x: 0 if abs(x) < 1e-4 else x)
        df['rebalance_date'] = rebalance_date
        
        # Standardize output columns
        return df[['rebalance_date', 'ticker', 'weight', 'delta_signal_smooth', 'industry']]

    def _fallback_heuristic_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        If CVXPY fails or isn't installed.
        Simple quintile portfolio: Long top 20%, Short bottom 20%.
        Equal-weight within legs. Not sector constrained.
        """
        n = len(df)
        alphas = df['delta_signal_smooth'].values
        
        # Determine quintile cutoffs
        top_quintile_thresh = np.percentile(alphas, 80)
        bot_quintile_thresh = np.percentile(alphas, 20)
        
        weights = np.zeros(n)
        
        long_idx = alphas >= top_quintile_thresh
        short_idx = alphas <= bot_quintile_thresh
        
        n_long = np.sum(long_idx)
        n_short = np.sum(short_idx)
        
        # Assign weights such that gross exposure = target_gross
        # and net exposure = 0
        leg_exposure = self.target_gross / 2.0
        
        if n_long > 0:
            weights[long_idx] = leg_exposure / n_long
        if n_short > 0:
            weights[short_idx] = -leg_exposure / n_short
            
        return weights
