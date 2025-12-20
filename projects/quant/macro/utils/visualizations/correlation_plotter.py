import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional

class CorrelationPlotter:
    
    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        figsize: tuple = (12, 10),
        ax: Optional[plt.Axes] = None,
        cmap: str = 'coolwarm',
        vmin: float = -1,
        vmax: float = 1
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            is_own_figure = True
        else:
            is_own_figure = False
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlación'},
            ax=ax
        )
        
        ax.set_title('Matriz de Correlación: Portfolio vs Factores Macro', 
                    fontsize=14, fontweight='bold')
        
        if is_own_figure:
            try:
                plt.tight_layout()
            except:
                pass
        
        return ax

    def plot_lagged_correlations(
        self,
        lagged_corrs: pd.DataFrame,
        top_n: Optional[int] = 10,
        figsize: tuple = (14, 8),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            is_own_figure = True
        else:
            is_own_figure = False
        
        corr_col = 'corr' if 'corr' in lagged_corrs.columns else 'correlation'

        lagged_corrs_sorted = lagged_corrs.sort_values(corr_col, key=abs, ascending=False)
        
        if top_n:
            lagged_corrs_sorted = lagged_corrs_sorted.head(top_n)
        
        factors = lagged_corrs_sorted['factor'].values
        correlations = lagged_corrs_sorted[corr_col].values
        lags = lagged_corrs_sorted['lag'].values

        colors = ['green' if c > 0 else 'red' for c in correlations]
        
        bars = ax.barh(factors, correlations, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Correlación', fontsize=11, fontweight='bold')
        ax.set_title(f'Top {top_n} Correlaciones con Lags', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, corr, lag) in enumerate(zip(bars, correlations, lags)):
            width = bar.get_width()
            label = f'{corr:.3f} (lag {int(lag)})'
            ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                label, ha='left' if width > 0 else 'right', va='center', fontsize=9)

        if is_own_figure:
            try:
                plt.tight_layout()
            except:
                pass
        
        return ax