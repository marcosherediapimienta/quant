import matplotlib.pyplot as plt
import numpy as np
from ...visualizations.components.capm_plotter import CAPMPlotter
from ...analysis.capm.analyzers.capm_analyzer import CAPMAnalyzer
from ...analysis.capm.components.helpers import daily_risk_free_rate


class CAPMVisualizer:

    def __init__(self, capm_analyzer: CAPMAnalyzer):
        self.analyzer = capm_analyzer
        self.plotter = CAPMPlotter()
    
    def plot_capm_analysis(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        risk_free_rate: float,
        asset_name: str = "Activo",
        figsize: tuple = (14, 10)
    ) -> plt.Figure:

        results = self.analyzer.analyze(asset_returns, market_returns, risk_free_rate)
        rf_daily = daily_risk_free_rate(risk_free_rate, self.analyzer.annual_factor)
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Regresión CAPM
        ax1 = fig.add_subplot(gs[0, :])
        self.plotter.plot_regression(
            market_returns,
            asset_returns,
            results['alpha_daily'],
            results['beta'],
            rf_daily,
            title=f"Regresión CAPM: {asset_name}",
            ax=ax1
        )
        
        # 2. Estadísticas (texto)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.axis('off')
        stats_text = f"""
        PARÁMETROS DEL MODELO
        
        Beta:              {results['beta']:.3f}
        Correlación:       {results['correlation']:.3f}
        R²:                {results['r_squared']:.3f}
        
        ALPHA (Jensen)
        
        Alpha Diario:      {results['alpha_daily']*100:.4f}%
        Alpha Anual:       {results['alpha_annual']*100:.2f}%
        
        SIGNIFICANCIA
        
        t-statistic:       {results['t_statistic']:.3f}
        p-value:           {results['p_value']:.4f}
        Significativo:      {'[SI]' if results['is_significant'] else '[NO]'}
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax2.transAxes)
        
        # 3. Distribución de residuos (si hay suficientes datos)
        ax3 = fig.add_subplot(gs[1, 1])
        market_excess = market_returns - rf_daily
        asset_excess = asset_returns - rf_daily
        predicted = results['alpha_daily'] + results['beta'] * market_excess
        residuals = asset_excess - predicted
        
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Residuos', fontsize=11)
        ax3.set_ylabel('Frecuencia', fontsize=11)
        ax3.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f"Análisis CAPM Completo: {asset_name}", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig