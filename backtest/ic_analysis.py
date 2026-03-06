"""
Information Coefficient (IC) Analysis.
Evaluates the predictive power of the divergence signal on forward estimate revisions.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import logging

try:
    import alphalens_reloaded as al
    ALPHALENS_AVAILABLE = True
except ImportError:
    ALPHALENS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class ICAnalyzer:
    def __init__(self, forward_windows=[1, 3, 5, 10, 15, 20, 30]):
        """
        Setup the IC Analyzer with specified trading day forward windows.
        """
        self.forward_windows = forward_windows

    def calculate_forward_revisions(self, ibes_df: pd.DataFrame, dates: list, tickers: list) -> pd.DataFrame:
        """
        Mock implementation.
        Computes net revision direction for each ticker over the N-day forward windows.
        Returns a DataFrame mapping [ticker, as_of_date] to {window_1_ret, window_3_ret, etc.}
        In reality, we want to know: "Did the consensus EPS/Rev estimate go UP or DOWN
        in the [as_of_date + N] days window?" 
        """
        logger.debug("Calculating forward revision targets for evaluation")
        # Mock logic
        records = []
        for d in dates:
            for t in tickers:
                rec = {"ticker": t, "date": d}
                for w in self.forward_windows:
                    # Random noise centered slightly around 0 representing forward revision direction
                    rec[f"fwd_rev_{w}d"] = np.random.normal(0, 0.05)
                records.append(rec)
                
        return pd.DataFrame(records)

    def run_ic_analysis(self, signals_df: pd.DataFrame, forward_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Spearman Rank IC between the daily cross-sectional signal
        and the N-day forward target.
        """
        if signals_df.empty or forward_df.empty:
            logger.warning("Empty dataframe passed to IC Analyzer.")
            return pd.DataFrame()

        # Merge signals and forward targets on ticker + date
        merged = pd.merge(signals_df, forward_df, on=['ticker', 'date'], how='inner')
        
        ic_results = []
        
        for window in self.forward_windows:
            target_col = f"fwd_rev_{window}d"
            if target_col not in merged.columns:
                continue
                
            # Compute daily cross-sectional Spearman IC
            def calc_spearman(group):
                x = group['delta_signal_smooth']
                y = group[target_col]
                # Drop NAs
                valid = ~(x.isna() | y.isna())
                if valid.sum() < 10:
                    return np.nan
                
                corr, _ = stats.spearmanr(x[valid], y[valid])
                return corr
                
            daily_ic = merged.groupby('date').apply(calc_spearman).dropna()
            
            if len(daily_ic) == 0:
                continue
                
            ic_mean = daily_ic.mean()
            ic_std = daily_ic.std()
            # IC Information Ratio (IC divided by standard deviation of IC) helps measure consistency
            icir = ic_mean / ic_std if ic_std != 0 else 0
            # t-stat for IC significance
            t_stat = ic_mean / (ic_std / np.sqrt(len(daily_ic))) if ic_std != 0 else 0
            
            ic_results.append({
                "Forward_Window": f"{window} Days",
                "IC_Mean": float(ic_mean),
                "IC_Std": float(ic_std),
                "ICIR": float(icir),
                "t_stat": float(t_stat),
                "Observations": int(len(daily_ic))
            })
            
        summary_df = pd.DataFrame(ic_results)
        logger.info(f"IC Analysis Results:\n{summary_df}")
        return summary_df
