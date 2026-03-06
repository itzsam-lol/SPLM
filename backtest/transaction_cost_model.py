"""
Transaction Cost Model (Indian Market).
Models NSE transaction costs including STT, SEBI, brokerage, etc.
Currency: INR (crores).
"""
import pandas as pd
import numpy as np

class TransactionCostModel:
    def __init__(self, broker_type='discount'):
        self.broker_type = broker_type

    def estimate_cost_india(self, trade_value_inr: float, broker_type: str = None) -> float:
        """
        trade_value_inr must be in \u20B9 crores or absolute \u20B9, return matches unit.
        """
        b_type = broker_type or self.broker_type
        brokerage = 0.0005 if b_type == 'discount' else 0.003
        
        COST_COMPONENTS = {
            'brokerage':   brokerage,
            'stt':         0.001,   # STT delivery
            'sebi_charge': 0.0001,  # SEBI charge
            'stamp_duty':  0.00015, # Stamp duty
            'gst_on_brok': 0.18 * brokerage,
        }
        
        total_rate = sum(COST_COMPONENTS.values())
        return total_rate * trade_value_inr

    def calculate_turnover_costs(self, w_current: pd.Series, w_target: pd.Series, aum_inr_cr: float) -> float:
        """
        Returns cost in INR crores based on portfolio turnover.
        """
        all_tickers = w_current.index.union(w_target.index)
        w_c = w_current.reindex(all_tickers).fillna(0)
        w_t = w_target.reindex(all_tickers).fillna(0)
        
        turnover_weights = np.abs(w_t - w_c)
        total_trade_value_cr = turnover_weights.sum() * aum_inr_cr
        
        return self.estimate_cost_india(total_trade_value_cr, self.broker_type)

    def apply_costs_to_returns(self, raw_returns: pd.Series, turnover: pd.Series, broker_type: str = 'discount') -> pd.Series:
        """
        Simplified application to daily/quarterly returns based on turnover series.
        """
        rate = self.estimate_cost_india(1.0, broker_type)
        return raw_returns - (turnover * rate)
