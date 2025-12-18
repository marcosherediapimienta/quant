import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ..components.helpers import calculate_portfolio_returns, annualize_return, annualize_volatility
from .var_es_analyzer import VarEsAnalyzer
from .ratio_analyzer import RatioAnalyzer
from .distribution_analyzer import DistributionAnalyzer


class ComprehensiveAnalyzer:

    def __init__(self, risk_analysis):
        self.risk_analysis = risk_analysis
        self.var_es = VarEsAnalyzer(risk_analysis)
        self.ratios = RatioAnalyzer(risk_analysis)
        self.distribution = DistributionAnalyzer(risk_analysis)
    
    def calculate_summary(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        risk_free_rate: float,
        confidence_levels: Tuple[float, ...] = (0.95, 0.99),
        var_method: str = 'historical',
        ddof: int = 0
    ) -> Dict[str, any]:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        # 1. Métricas básicas
        annual_return = annualize_return(portfolio_ret, self.risk_analysis.annual_factor)
        annual_vol = annualize_volatility(portfolio_ret, self.risk_analysis.annual_factor, ddof)
        
        # 2. Ratios
        ratio_results = self.ratios.calculate_all_ratios(
            returns, weights, risk_free_rate, ddof
        )
        
        # 3. Distribución
        dist_results = self.distribution.analyze(returns, weights)
        
        # 4. VaR y ES
        var_es_results = {}
        for cl in confidence_levels:
            var_result = self.risk_analysis.var.calculate(
                returns, weights, cl, var_method
            )
            es_result = self.risk_analysis.es.calculate(
                returns, weights, cl, var_method
            )
            
            var_es_results[f'VaR_{int(cl*100)}'] = var_result['var_daily']
            var_es_results[f'VaR_{int(cl*100)}_pct'] = var_result['var_daily_pct']
            var_es_results[f'ES_{int(cl*100)}'] = es_result['es_daily']
            var_es_results[f'ES_{int(cl*100)}_pct'] = es_result['es_daily_pct']
        
        summary = {
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_vol),
            **ratio_results,
            'skewness': dist_results['skewness'],
            'excess_kurtosis': dist_results['excess_kurtosis'],
            'is_normal_distribution': dist_results['is_normal'],
            'jarque_bera_pvalue': dist_results['jb_p_value'],
            **var_es_results,
            'n_observations': len(portfolio_ret),
        }
        
        return summary