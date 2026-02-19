import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional

class FactorLoadingsPlotter:
    
    def plot_loadings_bar(
        self,
        betas: Dict[str, float],
        significant: Optional[list] = None,
        figsize: tuple = (14, 8),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            is_own_figure = True
        else:
            is_own_figure = False

        sorted_betas = sorted(betas.items(), key=lambda x: abs(x[1]), reverse=True)
        factors = [f for f, _ in sorted_betas]
        values = [v for _, v in sorted_betas]

        colors = ['green' if v > 0 else 'red' for v in values]

        if significant:
            for i, factor in enumerate(factors):
                if factor in significant:
                    colors[i] = 'darkgreen' if values[i] > 0 else 'darkred'
        
        bars = ax.barh(factors, values, color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Beta (Sensitivity)', fontsize=11, fontweight='bold')
        ax.set_title('Factor Loadings (Betas) by Macro Factor', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for i, (bar, val) in enumerate(zip(bars, values)):
            width = bar.get_width()
            ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='left' if width > 0 else 'right', 
                va='center', fontsize=9)

        if is_own_figure:
            try:
                plt.tight_layout()
            except Exception:
                pass
        
        return ax

    def plot_loadings_comparison(
        self,
        betas_dict: Dict[str, Dict[str, float]],
        figsize: tuple = (14, 8),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            is_own_figure = True
        else:
            is_own_figure = False

        df = pd.DataFrame(betas_dict).T

        df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel('Beta', fontsize=11, fontweight='bold')
        ax.set_title('Factor Loadings Comparison', fontsize=14, fontweight='bold')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.legend(title='Period', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        if is_own_figure:
            try:
                plt.tight_layout()
            except Exception:
                pass
        
        return ax