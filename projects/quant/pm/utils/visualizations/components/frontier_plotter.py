import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from ...analysis.capm.components.efficient_frontier import FrontierResult
from ...analysis.capm.components.cml_calculator import CMLResult
from ...analysis.capm.components.sml_calculator import SMLResult


class FrontierPlotter:

    def plot_efficient_frontier(
        self,
        frontier: FrontierResult,
        title: str = "Frontera Eficiente",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(
            frontier.volatilities * 100,
            frontier.returns * 100,
            'b-',
            linewidth=2,
            label='Frontera Eficiente'
        )
        
        ax.set_xlabel('Volatilidad (%)', fontsize=12)
        ax.set_ylabel('Retorno Esperado (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_frontier_with_cml(
        self,
        frontier: FrontierResult,
        cml: CMLResult,
        risk_free_rate: float,
        title: str = "Frontera Eficiente y CML",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Frontera eficiente
        ax.plot(
            frontier.volatilities * 100,
            frontier.returns * 100,
            'b-',
            linewidth=2,
            label='Frontera Eficiente'
        )
        
        # CML
        ax.plot(
            cml.cml_volatilities * 100,
            cml.cml_returns * 100,
            'r--',
            linewidth=2,
            label='Capital Market Line (CML)'
        )
        
        # Portafolio tangente
        if not np.isnan(cml.tangent_return):
            ax.scatter(
                cml.tangent_volatility * 100,
                cml.tangent_return * 100,
                s=200,
                color='red',
                marker='*',
                zorder=5,
                label=f'Portafolio Tangente\n(Sharpe={cml.slope:.3f})',
                edgecolors='black',
                linewidth=1.5
            )
        
        # Tasa libre de riesgo
        ax.scatter(
            0,
            risk_free_rate * 100,
            s=150,
            color='green',
            marker='o',
            zorder=5,
            label=f'Tasa Libre de Riesgo ({risk_free_rate*100:.2f}%)',
            edgecolors='black',
            linewidth=1.5
        )
        
        ax.set_xlabel('Volatilidad (%)', fontsize=12)
        ax.set_ylabel('Retorno Esperado (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_sml(
        self,
        sml: SMLResult,
        asset_betas: Optional[dict] = None,
        asset_returns: Optional[dict] = None,
        title: str = "Security Market Line (SML)",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # SML
        ax.plot(
            sml.beta_axis,
            sml.expected_returns * 100,
            'b-',
            linewidth=2,
            label=f'SML (Pendiente={sml.slope*100:.2f}%)'
        )
        
        # Activos si se proporcionan
        if asset_betas is not None and asset_returns is not None:
            for asset, beta in asset_betas.items():
                expected = sml.risk_free_rate + sml.slope * beta
                actual = asset_returns.get(asset, expected)
                
                # Color según si está sobre/under valorado
                color = 'green' if actual > expected else 'red'
                marker = '^' if actual > expected else 'v'
                
                ax.scatter(
                    beta,
                    actual * 100,
                    s=150,
                    color=color,
                    marker=marker,
                    zorder=5,
                    edgecolors='black',
                    linewidth=1.5
                )
                
                # Línea vertical al SML
                ax.plot(
                    [beta, beta],
                    [expected * 100, actual * 100],
                    'k--',
                    alpha=0.3,
                    linewidth=1
                )
                
                ax.annotate(
                    asset,
                    (beta, actual * 100),
                    fontsize=9,
                    alpha=0.8,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
        
        # Marcadores de referencia
        ax.axvline(x=1, color='k', linestyle='--', linewidth=1, alpha=0.5, label='β=1 (Mercado)')
        ax.scatter(
            1,
            sml.market_return * 100,
            s=200,
            color='blue',
            marker='*',
            zorder=5,
            label='Mercado',
            edgecolors='black',
            linewidth=1.5
        )
        
        ax.set_xlabel('Beta', fontsize=12)
        ax.set_ylabel('Retorno Esperado (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax