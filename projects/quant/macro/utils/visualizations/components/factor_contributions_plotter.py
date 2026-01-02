import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

class FactorContributionsPlotter:
    
    def plot_accumulated_contributions(
        self,
        contributions: pd.DataFrame,
        top_n: Optional[int] = None,
        exclude: Optional[list] = None,
        figsize: tuple = (16, 8),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        
        if exclude is None:
            exclude = ['alpha', 'residual']
        
        plot_cols = [col for col in contributions.columns 
                    if col not in exclude]

        contrib_clean = contributions[plot_cols].fillna(0).replace([np.inf, -np.inf], 0)
        contrib_clean = contrib_clean.clip(lower=-0.1, upper=0.1)

        contrib_cumsum = (1 + contrib_clean).cumprod()

        valid_cols = []
        for col in contrib_cumsum.columns:
            col_data = contrib_cumsum[col].dropna()
            if len(col_data) > 0:
                max_abs = col_data.abs().max()
                if 0.01 < max_abs < 100: 
                    valid_cols.append((col, max_abs))

        valid_cols.sort(key=lambda x: x[1], reverse=True)
        if top_n:
            valid_cols = valid_cols[:top_n]
        
        plot_cols = [col for col, _ in valid_cols]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        for col in plot_cols:
            ax.plot(contrib_cumsum.index, contrib_cumsum[col], 
                   label=col, linewidth=2, alpha=0.8)
        
        ax.set_title('Contribución Acumulada por Factor Macro', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Crecimiento (base 1)')
        ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(loc='best', ncol=2, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        return ax