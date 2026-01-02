import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
from ...analyzers.macro_correlation_analyzer import MacroCorrelationAnalyzer
from ..components.correlation_plotter import CorrelationPlotter

class MacroCorrelationVisualizer:
    
    def __init__(self, analyzer: MacroCorrelationAnalyzer):
        self.analyzer = analyzer
        self.correlation_plotter = CorrelationPlotter()
    
    def plot_correlation_analysis(
        self,
        correlation_results: Dict,
        top_n: int = 10,
        figsize: tuple = (16, 10)
    ) -> plt.Figure:
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        best_lags = correlation_results.get('best_lagged_correlations', pd.DataFrame())
        if not best_lags.empty:
            self.correlation_plotter.plot_lagged_correlations(
                best_lags, top_n=top_n, ax=ax1
            )

        if 'correlation_matrix' in correlation_results:
            ax2 = fig.add_subplot(gs[1, 0])
            corr_matrix = correlation_results['correlation_matrix']
            if isinstance(corr_matrix, dict):
                if 0 in corr_matrix:
                    corr_matrix = corr_matrix[0]
                else:
                    corr_matrix = list(corr_matrix.values())[0]
            
            if isinstance(corr_matrix, pd.DataFrame):
                self.correlation_plotter.plot_correlation_heatmap(
                    corr_matrix, ax=ax2
                )
        
        plt.suptitle('Análisis de Correlación: Portfolio vs Factores Macro',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_lagged_correlations(
        self,
        correlation_results: Dict,
        top_n: int = 10,
        figsize: tuple = (14, 8)
    ) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        best_lags = correlation_results.get('best_lagged_correlations', pd.DataFrame())
        if not best_lags.empty:
            self.correlation_plotter.plot_lagged_correlations(
                best_lags, top_n=top_n, ax=ax
            )
        return fig