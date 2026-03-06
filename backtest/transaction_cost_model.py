"""
Transaction Cost Model.
Estimates the execution slippage and flat fees for portfolio turnover.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TransactionCostModel:
    def __init__(self, flat_bps: float = 5.0, adv_penalty_factor: float = 0.1):
        """
        flat_bps: Base cost for trading large cap liquid names (e.g. 5 bps = 0.05%)
        adv_penalty_factor: Multiplier to penalize trades that consume high % of ADV.
        """
        self.flat_bps = flat_bps
        self.adv_penalty_factor = adv_penalty_factor

    def estimate_costs(self, current_weights: pd.Series, target_weights: pd.Series, portfolio_aum: float, daily_adv_data: pd.DataFrame = None) -> float:
        """
        Calculate the transaction cost in dollars for rebalancing from current to target weights.
        """
        # Align indices (tickers)
        all_tickers = current_weights.index.union(target_weights.index)
        
        w_curr = current_weights.reindex(all_tickers).fillna(0)
        w_targ = target_weights.reindex(all_tickers).fillna(0)
        
        # Turnover in weight percentages
        turnover_weights = np.abs(w_targ - w_curr)
        
        # Base flat fee
        base_cost_dollars = turnover_weights.sum() * portfolio_aum * (self.flat_bps / 10000.0)
        
        # Impact penalty
        impact_cost_dollars = 0.0
        if daily_adv_data is not None and not daily_adv_data.empty:
            for ticker in turnover_weights.index[turnover_weights > 0]:
                if ticker in daily_adv_data.index:
                    trade_size_dollars = turnover_weights[ticker] * portfolio_aum
                    adv_dollars = daily_adv_data.loc[ticker, 'adv_30d']
                    
                    if adv_dollars > 0:
                        # Simple linear market impact model:
                        # penalty = factor * (TradeSize / ADV)
                        pct_of_adv = trade_size_dollars / adv_dollars
                        impact_rate = self.adv_penalty_factor * pct_of_adv
                        impact_cost_dollars += trade_size_dollars * impact_rate
                        
        total_costs = base_cost_dollars + impact_cost_dollars
        
        return total_costs

    def apply_costs_to_returns(self, raw_returns: pd.Series, turnover: pd.Series, avg_bps: float = 7.5) -> pd.Series:
        """
        Simpler macro-level application of costs to daily returns.
        raw_returns: Daily portfolio returns
        turnover: Daily portfolio turnover (sum of abs weight changes / 2)
        """
        cost_rate = avg_bps / 10000.0
        net_returns = raw_returns - (turnover * cost_rate)
        return net_returns
