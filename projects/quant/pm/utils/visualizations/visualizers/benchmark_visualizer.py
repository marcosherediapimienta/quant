import matplotlib.pyplot as plt
import pandas as pd
from ..components.rolling_plotter import RollingPlotter
from ...tools.config import ANNUAL_FACTOR

class BenchmarkVisualizer:
    def __init__(self, annual_factor: float = None):
        self.annual_factor = annual_factor or ANNUAL_FACTOR
        self.rolling_plotter = RollingPlotter()
    
    def plot_rolling_benchmark(
        self,
        rolling_data: pd.DataFrame,
        figsize: tuple = (14, 8)
    ) -> plt.Figure:

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        self.rolling_plotter.plot_rolling_ratio(
            rolling_data['tracking_error'],
            title='Tracking Error Rolling',
            ylabel='Tracking Error',
            ax=axes[0]
        )

        self.rolling_plotter.plot_rolling_ratio(
            rolling_data['beta'],
            title='Beta Rolling',
            ylabel='Beta',
            ax=axes[1]
        )
        
        plt.tight_layout()
        return fig