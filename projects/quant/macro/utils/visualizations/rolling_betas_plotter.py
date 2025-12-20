import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

class RollingBetasPlotter:
    
    def plot_rolling_betas(
        self,
        rolling_betas: pd.DataFrame,
        factors: Optional[list] = None,
        figsize: tuple = (16, 10),
        n_cols: int = 2
    ) -> plt.Figure:
        
        beta_cols = [col for col in rolling_betas.columns if col.startswith('beta_')]
        
        if factors:
            beta_cols = [col for col in beta_cols if any(f in col for f in factors)]
        
        n_factors = len(beta_cols)
        n_rows = (n_factors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_factors == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(beta_cols[:len(axes)]):
            factor_name = col.replace('beta_', '')
            axes[i].plot(rolling_betas.index, rolling_betas[col], linewidth=2)
            axes[i].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[i].set_title(f'Beta Rolling: {factor_name}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Fecha')
            axes[i].set_ylabel('Beta')
            axes[i].grid(True, alpha=0.3)
 
        for i in range(len(beta_cols), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Evolución de Betas (Rolling Window)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig