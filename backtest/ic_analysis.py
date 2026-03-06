"""
Information Coefficient (IC) Analysis.
Evaluates predictive power of divergence signal on forward revenue surprise.
Horizons in quarters.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

FORWARD_HORIZONS = [1, 2, 3, 4]  # quarters

class ICAnalyzer:
    def __init__(self, forward_windows=FORWARD_HORIZONS):
        self.forward_windows = forward_windows

    def calculate_forward_targets(self, targets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates forward-looking target variables for N quarters ahead.
        targets_df should have [ticker, year, quarter, revenue_surprise_zscore]
        """
        if targets_df.empty: return pd.DataFrame()
        df = targets_df.sort_values(['ticker', 'year', 'quarter']).copy()
        
        for w in self.forward_windows:
            df[f'fwd_rev_{w}q'] = df.groupby('ticker')['revenue_surprise_zscore'].shift(-w)
            
        return df

    def run_ic_analysis(self, signals_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Spearman Rank IC between cross-sectional signal
        and the N-quarter forward target.
        """
        if signals_df.empty or target_df.empty:
            return pd.DataFrame()

        merged = pd.merge(signals_df, target_df, on=['ticker', 'year', 'quarter'], how='inner')
        ic_results = []
        
        for w in self.forward_windows:
            t_col = f'fwd_rev_{w}q'
            if t_col not in merged.columns: continue
            
            def calc_spearman(group):
                x = group['delta_signal_smooth']
                y = group[t_col]
                valid = ~(x.isna() | y.isna())
                if valid.sum() < 5: return np.nan
                corr, _ = stats.spearmanr(x[valid], y[valid])
                return corr
                
            quarterly_ic = merged.groupby(['year', 'quarter']).apply(calc_spearman).dropna()
            
            if len(quarterly_ic) == 0: continue
            
            ic_mean = quarterly_ic.mean()
            ic_std = quarterly_ic.std()
            icir = ic_mean / ic_std if ic_std != 0 else 0
            t_stat = ic_mean / (ic_std / np.sqrt(len(quarterly_ic))) if len(quarterly_ic) > 0 and ic_std != 0 else 0
            hit_rate = (quarterly_ic > 0).mean()
            
            ic_results.append({
                "Forward_Window": f"Q+{w}",
                "IC_Mean": float(ic_mean),
                "IC_Std": float(ic_std),
                "ICIR": float(icir),
                "t_stat": float(t_stat),
                "Hit_Rate": float(hit_rate),
                "Observations": int(len(quarterly_ic))
            })
            
        summary_df = pd.DataFrame(ic_results)
        logger.info(f"IC Analysis:\n{summary_df}")
        return summary_df
