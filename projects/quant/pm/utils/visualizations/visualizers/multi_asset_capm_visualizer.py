import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ...visualizations.components.capm_plotter import CAPMPlotter
from ...analysis.capm.analyzers.multi_asset_capm_analyzer import MultiAssetCAPMAnalyzer


class MultiAssetCAPMVisualizer:

    def __init__(self, multi_asset_analyzer: MultiAssetCAPMAnalyzer):
        self.analyzer = multi_asset_analyzer
        self.plotter = CAPMPlotter()
    
    def plot_multi_asset_analysis(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        risk_free_rate: float,
        figsize: tuple = (16, 10)
    ) -> plt.Figure:

        results_df = self.analyzer.analyze_multiple(returns, market_returns, risk_free_rate)
        
        if results_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No hay datos suficientes', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Alpha vs Beta
        ax1 = fig.add_subplot(gs[0, :])
        self.plotter.plot_alpha_beta_comparison(
            results_df,
            title="Alpha vs Beta - Comparación Multi-Activo",
            ax=ax1
        )
        
        # 2. Top 10 Alphas
        ax2 = fig.add_subplot(gs[1, 0])
        top10 = results_df.nlargest(10, 'alpha_annual')
        colors = ['green' if sig else 'gray' for sig in top10['is_significant']]
        ax2.barh(range(len(top10)), top10['alpha_annual'] * 100, color=colors)
        ax2.set_yticks(range(len(top10)))
        ax2.set_yticklabels(top10.index)
        ax2.set_xlabel('Alpha Anual (%)', fontsize=11)
        ax2.set_title('Top 10 Alphas', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Distribución de Betas
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(results_df['beta'], bins=15, alpha=0.7, edgecolor='black')
        ax3.axvline(x=1, color='r', linestyle='--', linewidth=2, label='β=1 (Mercado)')
        ax3.axvline(x=results_df['beta'].mean(), color='g', linestyle='--', 
                   linewidth=2, label=f'Media: {results_df["beta"].mean():.2f}')
        ax3.set_xlabel('Beta', fontsize=11)
        ax3.set_ylabel('Frecuencia', fontsize=11)
        ax3.set_title('Distribución de Betas', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle("Análisis CAPM Multi-Activo", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig