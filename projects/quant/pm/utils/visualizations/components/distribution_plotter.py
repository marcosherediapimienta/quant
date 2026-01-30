import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Tuple

class DistributionPlotter:
    def __init__(self, figsize: Tuple[int, int] = (12, 5)):
        self.figsize = figsize
    
    def plot_histogram(
        self,
        returns: pd.Series,
        title: str = "Distribución de Retornos",
        bins: int = 50,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        ax.hist(returns, bins=bins, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black', label='Retornos')

        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        normal = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal, 'r-', linewidth=2, label='Normal teórica')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Retorno', fontsize=12)
        ax.set_ylabel('Densidad', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_qq(
        self,
        returns: pd.Series,
        title: str = "Q-Q Plot",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        stats.probplot(returns, dist="norm", plot=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return ax