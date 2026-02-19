import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict
from ..components.distribution_plotter import DistributionPlotter
from ...analysis.risk_metrics.components.helpers import calculate_portfolio_returns

class DistributionVisualizer:
    def __init__(self):
        self.dist_plotter = DistributionPlotter()
    
    def plot_distribution_analysis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        dist_results: Dict[str, float],
        figsize: tuple = (14, 6)
    ) -> plt.Figure:

        portfolio_ret = calculate_portfolio_returns(returns, weights)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        self.dist_plotter.plot_histogram(
            portfolio_ret,
            title='Portfolio Returns Distribution',
            ax=axes[0]
        )
        
        stats_text = (
            f"Mean: {dist_results['mean']*100:.2f}%\n"
            f"Std: {dist_results['std']*100:.2f}%\n"
            f"Skewness: {dist_results['skewness']:.3f}\n"
            f"Kurtosis: {dist_results['excess_kurtosis']:.3f}\n"
            f"JB p-value: {dist_results['jb_p_value']:.4f}"
        )
        axes[0].text(0.98, 0.98, stats_text,
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10)

        self.dist_plotter.plot_qq(
            portfolio_ret,
            title='Q-Q Plot (Normality Test)',
            ax=axes[1]
        )
        
        plt.tight_layout()
        return fig