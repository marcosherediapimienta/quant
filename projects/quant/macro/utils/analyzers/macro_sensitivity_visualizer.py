import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional
from ..analyzers.macro_sensitivity_analyzer import MacroSensitivityAnalyzer
from ..visualizations.rolling_betas_plotter import RollingBetasPlotter
from ..visualizations.factor_loadings_plotter import FactorLoadingsPlotter

class MacroSensitivityVisualizer:
    
    def __init__(self, analyzer: MacroSensitivityAnalyzer):
        self.analyzer = analyzer
        self.rolling_plotter = RollingBetasPlotter()
        self.loadings_plotter = FactorLoadingsPlotter()
    
    def plot_sensitivity_analysis(
        self,
        sensitivity_results: Dict,
        rolling_betas: Optional[pd.DataFrame] = None,
        figsize: tuple = (16, 12)
    ) -> plt.Figure:
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, hspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        betas = sensitivity_results.get('betas', {})
        if betas:
            self.loadings_plotter.plot_loadings_bar(betas, ax=ax1)
            ax1.set_title('Exposiciones Actuales (Betas)', fontsize=12, fontweight='bold')

        if rolling_betas is not None and not rolling_betas.empty:
            ax2 = fig.add_subplot(gs[1, 0])

            if betas:
                top_factors = sorted(betas.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
                factor_names = [f.replace('beta_', '') for f, _ in top_factors]
                beta_cols = [f'beta_{f}' for f in factor_names if f'beta_{f}' in rolling_betas.columns]
                
                for col in beta_cols[:6]:
                    factor_name = col.replace('beta_', '')
                    ax2.plot(rolling_betas.index, rolling_betas[col], 
                           label=factor_name, linewidth=2, alpha=0.8)
                
                ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax2.set_title('Evolución de Exposiciones (Rolling Betas)', 
                             fontsize=12, fontweight='bold')
                ax2.set_xlabel('Fecha')
                ax2.set_ylabel('Beta')
                ax2.legend(loc='best', ncol=2)
                ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Análisis de Sensibilidad a Factores Macro',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_rolling_betas(
        self,
        rolling_betas: pd.DataFrame,
        factors: Optional[list] = None,
        figsize: tuple = (16, 10)
    ) -> plt.Figure:
        
        return self.rolling_plotter.plot_rolling_betas(
            rolling_betas, factors=factors, figsize=figsize
        )