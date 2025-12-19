import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


class CAPMPlotter:

    def plot_regression(
        self,
        market_returns: np.ndarray,
        asset_returns: np.ndarray,
        alpha: float,
        beta: float,
        risk_free_rate: float,
        title: str = "Regresión CAPM",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Excess returns
        market_excess = market_returns - risk_free_rate
        asset_excess = asset_returns - risk_free_rate
        
        # Scatter plot
        ax.scatter(market_excess, asset_excess, alpha=0.5, s=20, label='Observaciones')
        
        # Línea de regresión
        x_line = np.linspace(market_excess.min(), market_excess.max(), 100)
        y_line = alpha + beta * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'CAPM: α={alpha:.4f}, β={beta:.3f}')
        
        # Línea de mercado (beta=1)
        ax.plot(x_line, x_line, 'k--', linewidth=1, alpha=0.5, label='Mercado (β=1)')
        
        ax.set_xlabel('Retorno Exceso del Mercado', fontsize=12)
        ax.set_ylabel('Retorno Exceso del Activo', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_alpha_beta_comparison(
        self,
        results_df: 'pd.DataFrame',
        title: str = "Comparación Alpha vs Beta",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot
        ax.scatter(
            results_df['beta'],
            results_df['alpha_annual'] * 100,
            s=100,
            alpha=0.6,
            c=results_df['alpha_annual'] * 100,
            cmap='RdYlGn',
            edgecolors='black',
            linewidth=1
        )
        
        # Etiquetas
        for asset in results_df.index:
            ax.annotate(
                asset,
                (results_df.loc[asset, 'beta'], results_df.loc[asset, 'alpha_annual'] * 100),
                fontsize=9,
                alpha=0.8
            )
        
        # Líneas de referencia
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=1, color='k', linestyle='--', linewidth=1, alpha=0.5, label='β=1')
        
        ax.set_xlabel('Beta', fontsize=12)
        ax.set_ylabel('Alpha Anual (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax