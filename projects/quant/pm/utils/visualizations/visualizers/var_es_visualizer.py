import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict
from ..components.var_es_plotter import VarEsPlotter
from ...analysis.risk_metrics.components.helpers import calculate_portfolio_returns

class VarEsVisualizer:
    
    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.var_es_plotter = VarEsPlotter()
    
    def plot_var_es_analysis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        var_es_results: Dict[str, Dict[str, Dict[str, float]]],
        confidence_level: float = 0.95,
        figsize: tuple = (14, 10)
    ) -> plt.Figure:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        methods_results = var_es_results[confidence_level]

        ax1 = fig.add_subplot(gs[0, 0])
        self.var_es_plotter.plot_var_es_comparison(
            portfolio_ret,
            methods_results,
            confidence_level=confidence_level,
            title='VaR y ES Diarios',
            ax=ax1
        )

        ax2 = fig.add_subplot(gs[0, 1])
        var_annual_results = {
            method: {
                'var_daily_pct': results['var_annual_pct'],
                'es_daily_pct': results['es_annual_pct']
            }
            for method, results in methods_results.items()
        }
        self.var_es_plotter.plot_var_es_comparison(
            portfolio_ret,
            var_annual_results,
            confidence_level=confidence_level,
            title='VaR y ES Anualizados',
            ax=ax2
        )

        methods = ['historical', 'parametric', 'monte_carlo']
        for idx, method in enumerate(methods):
            ax = fig.add_subplot(gs[1 + idx // 2, idx % 2])
            var_threshold = methods_results[method]['var_daily_pct']
            self.var_es_plotter.plot_var_breach_analysis(
                portfolio_ret,
                var_threshold,
                title=f'Violaciones VaR - {method.capitalize()}',
                ax=ax
            )
        
        plt.suptitle(f'Análisis VaR y ES (Confianza: {confidence_level*100:.0f}%)',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig