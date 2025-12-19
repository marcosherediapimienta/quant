import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple

class DrawdownPlotter:
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
    
    def plot_drawdown(
        self,
        cumulative_returns: pd.Series,
        drawdown_series: pd.Series,
        title: str = "Análisis de Drawdown",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Retornos acumulados
        ax.plot(cumulative_returns.index, cumulative_returns.values, 
                linewidth=1.5, label='Retornos Acumulados', color='steelblue')
        
        # Drawdown (área sombreada)
        ax.fill_between(drawdown_series.index, 0, drawdown_series.values,
                        alpha=0.3, color='red', label='Drawdown')
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Retorno Acumulado / Drawdown', fontsize=12)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax