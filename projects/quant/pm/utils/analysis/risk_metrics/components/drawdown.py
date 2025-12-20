import numpy as np
import pandas as pd
from typing import Dict
from .helpers import calculate_portfolio_returns, annualize_return

class DrawdownCalculator:

    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
    
    def calculate(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> Dict:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        cumulative = (1 + portfolio_ret).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1.0
        max_dd = float(drawdown.min())
        max_dd_date = drawdown.idxmin()
        underwater = drawdown < 0
        current_duration = 0
        max_duration = 0
        in_drawdown = False

        for date, is_underwater in underwater.items():
            if is_underwater:
                if not in_drawdown:
                    in_drawdown = True
                    drawdown_start = date
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                in_drawdown = False
                current_duration = 0

        annual_return = annualize_return(portfolio_ret, self.annual_factor)
        calmar = float(annual_return / abs(max_dd)) if max_dd < 0 else np.nan
        dd_monthly = drawdown.resample('ME').min()
        worst_3 = dd_monthly.nsmallest(3)
        
        if len(worst_3) >= 1 and worst_3.mean() < 0:
            sterling = float((annual_return - risk_free_rate) / abs(worst_3.mean()))
        else:
            sterling = np.nan
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'max_drawdown_date': max_dd_date,
            'max_underwater_duration': int(max_duration),
            'calmar_ratio': calmar,
            'sterling_ratio': sterling,
            'drawdown_series': drawdown,
            'cumulative_returns': cumulative,
            'annual_return': float(annual_return)
        }