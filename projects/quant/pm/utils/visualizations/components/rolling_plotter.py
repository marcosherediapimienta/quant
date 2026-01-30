import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple

class RollingPlotter:
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
    
    def plot_rolling_ratio(
        self,
        data: pd.Series,
        title: str,
        ylabel: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(data.index, data.values, linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend([ylabel])
        
        return ax
    
    def plot_multiple_rolling(
        self,
        data: pd.DataFrame,
        title: str,
        ylabel: str,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        for col in data.columns:
            ax.plot(data.index, data[col].values, label=col, linewidth=1.5)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax