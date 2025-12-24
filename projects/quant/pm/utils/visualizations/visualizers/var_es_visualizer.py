import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict
from ..components.var_es_plotter import VarEsPlotter
from ...analysis.risk_metrics.components.helpers import calculate_portfolio_returns
from ...tools.config import ANNUAL_FACTOR, DEFAULT_CONFIDENCE_LEVEL

class VarEsVisualizer:
    """
    Visualizer para gráficos de VaR y ES.
    
    Responsabilidad: Generar visualizaciones de VaR y ES.
    """
    
    def __init__(self, annual_factor: float = None):
        """
        Args:
            annual_factor: Factor de anualización. Por defecto usa config.ANNUAL_FACTOR
        """
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.var_es_plotter = VarEsPlotter()
    
    def plot_var_es_analysis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        var_es_results: Dict[str, Dict[str, Dict[str, float]]],
        confidence_level: float = None,
        figsize: tuple = (14, 10)
    ) -> plt.Figure:
        """
        Genera análisis visual completo de VaR y ES.
        
        Args:
            returns: DataFrame de retornos diarios
            weights: Array de pesos del portafolio
            var_es_results: Resultados de VaR y ES por nivel de confianza y método
            confidence_level: Nivel de confianza. Por defecto usa config
            figsize: Tamaño de la figura
            
        Returns:
            Figura de matplotlib con los gráficos
        """
        if confidence_level is None:
            confidence_level = DEFAULT_CONFIDENCE_LEVEL

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