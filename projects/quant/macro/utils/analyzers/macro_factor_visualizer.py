import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional
from ..analyzers.macro_factor_analyzer import MacroFactorAnalyzer
from ..visualizations.factor_contributions_plotter import FactorContributionsPlotter
from ..visualizations.factor_loadings_plotter import FactorLoadingsPlotter

class MacroFactorVisualizer:
    
    def __init__(self, analyzer: MacroFactorAnalyzer):
        self.analyzer = analyzer
        self.contributions_plotter = FactorContributionsPlotter()
        self.loadings_plotter = FactorLoadingsPlotter()
    
    def plot_factor_analysis(
        self,
        factor_results: Dict,
        top_n: Optional[int] = 10,
        figsize: tuple = (16, 12)
    ) -> plt.Figure:
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        contributions = pd.DataFrame(factor_results['factor_contributions'])
        self.contributions_plotter.plot_accumulated_contributions(
            contributions, top_n=top_n, ax=ax1
        )

        ax2 = fig.add_subplot(gs[1, 0])
        betas = factor_results['betas']
        significant = factor_results.get('significant_factors', [])
        self.loadings_plotter.plot_loadings_bar(
            betas, significant=significant, ax=ax2
        )

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')

        risk_decomp = factor_results.get('risk_decomposition', {})
        systematic_pct = risk_decomp.get('systematic_pct', risk_decomp.get('systematic_risk', 0))
        idiosyncratic_pct = risk_decomp.get('idiosyncratic_pct', risk_decomp.get('idiosyncratic_risk', 0))

        stats_text = f"""
        ESTADÍSTICAS DEL MODELO

        R²:                {factor_results['r_squared']:.4f}
        R² Ajustado:       {factor_results['adj_r_squared']:.4f}
        Observaciones:     {factor_results['n_obs']}

        ALPHA

        Alpha Diario:      {factor_results['alpha']*100:.4f}%
        Alpha Anual:       {factor_results['alpha_annual']*100:.2f}%

        RIESGO

        Sistemático:       {systematic_pct:.2f}%
        Idiosincrático:    {idiosyncratic_pct:.2f}%

        FACTORES SIGNIFICATIVOS

        {len(significant)} de {len(betas)} factores
        """

        ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax3.transAxes)
        plt.suptitle('Análisis de Factores Macro - Resumen Completo',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_contributions(
        self,
        factor_results: Dict,
        top_n: Optional[int] = None,
        figsize: tuple = (16, 8)
    ) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        contributions = pd.DataFrame(factor_results['factor_contributions'])
        self.contributions_plotter.plot_accumulated_contributions(
            contributions, top_n=top_n, ax=ax
        )
        return fig
    
    def plot_loadings(
        self,
        factor_results: Dict,
        figsize: tuple = (14, 8)
    ) -> plt.Figure:
        
        fig, ax = plt.subplots(figsize=figsize)
        betas = factor_results['betas']
        significant = factor_results.get('significant_factors', [])
        self.loadings_plotter.plot_loadings_bar(
            betas, significant=significant, ax=ax
        )
        return fig