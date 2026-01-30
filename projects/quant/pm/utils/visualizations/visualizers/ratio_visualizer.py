import matplotlib.pyplot as plt
import pandas as pd
from ..components.rolling_plotter import RollingPlotter
from ...tools.config import ANNUAL_FACTOR

class RatioVisualizer:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.rolling_plotter = RollingPlotter()
    
    def plot_rolling_ratios(
        self,
        rolling_data: pd.DataFrame,
        figsize: tuple = (14, 8)
    ) -> plt.Figure:

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        self.rolling_plotter.plot_rolling_ratio(
            rolling_data['sharpe_rolling'],
            title='Sharpe Ratio Rolling',
            ylabel='Sharpe Ratio',
            ax=axes[0]
        )
    
        self.rolling_plotter.plot_rolling_ratio(
            rolling_data['sortino_rolling'],
            title='Sortino Ratio Rolling',
            ylabel='Sortino Ratio',
            ax=axes[1]
        )
        
        plt.tight_layout()
        return fig