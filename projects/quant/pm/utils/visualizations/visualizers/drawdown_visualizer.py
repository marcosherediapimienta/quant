import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict
from ..components.drawdown_plotter import DrawdownPlotter
from ...analysis.risk_metrics.components.helpers import calculate_portfolio_returns

class DrawdownVisualizer:
    
    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.dd_plotter = DrawdownPlotter()
    
    def plot_drawdown_analysis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        dd_results: Dict[str, float],
        figsize: tuple = (14, 8)
    ) -> plt.Figure:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        cumulative_returns = (1 + portfolio_ret).cumprod() - 1
        running_max = cumulative_returns.cummax()
        drawdown = cumulative_returns - running_max
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        axes[0].plot(cumulative_returns.index, cumulative_returns.values,
                    linewidth=1.5, label='Retornos Acumulados', color='steelblue')
        axes[0].plot(running_max.index, running_max.values,
                    linewidth=1, linestyle='--', label='Máximo Histórico', 
                    color='green', alpha=0.7)
        axes[0].set_title('Evolución del Portfolio', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Retorno Acumulado', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        max_dd_date = dd_results['max_drawdown_date']
        max_dd_value = drawdown.loc[max_dd_date]
        axes[0].scatter([max_dd_date], [cumulative_returns.loc[max_dd_date]], 
                       color='red', s=100, zorder=5, 
                       label=f"Max DD: {dd_results['max_drawdown_pct']:.2f}%")
        axes[0].legend()
        axes[1].fill_between(drawdown.index, 0, drawdown.values * 100,
                            alpha=0.3, color='red')
        axes[1].plot(drawdown.index, drawdown.values * 100,
                    linewidth=1.5, color='darkred')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].set_xlabel('Fecha', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=dd_results['max_drawdown_pct'], 
                       color='red', linestyle='--', alpha=0.7,
                       label=f"Max DD: {dd_results['max_drawdown_pct']:.2f}%")
        axes[1].legend()

        stats_text = (
            f"Max Drawdown: {dd_results['max_drawdown_pct']:.2f}%\n"
            f"Fecha: {dd_results['max_drawdown_date'].date()}\n"
            f"Duración: {dd_results['max_underwater_duration']} días\n"
            f"Calmar Ratio: {dd_results['calmar_ratio']:.3f}\n"
            f"Sterling Ratio: {dd_results['sterling_ratio']:.3f}"
        )
        axes[1].text(0.02, 0.02, stats_text,
                    transform=axes[1].transAxes,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)
        
        plt.tight_layout()
        return fig