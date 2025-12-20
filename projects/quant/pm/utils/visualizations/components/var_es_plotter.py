import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class VarEsPlotter:
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
    
    def plot_var_es_comparison(
        self,
        returns: pd.Series,
        var_results: Dict[str, Dict[str, float]],
        confidence_level: float = 0.95,
        title: str = "Comparación VaR y ES por Método",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(var_results.keys())
        var_values = [var_results[m]['var_daily_pct'] for m in methods]
        es_values = [var_results[m]['es_daily_pct'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, var_values, width, label='VaR', alpha=0.8)
        bars2 = ax.bar(x + width/2, es_values, width, label='ES', alpha=0.8)
        
        ax.set_title(f'{title} (Confianza: {confidence_level*100:.0f}%)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Pérdida (%)', fontsize=12)
        ax.set_xlabel('Método', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in methods])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        return ax
    
    def plot_var_breach_analysis(
        self,
        returns: pd.Series,
        var_threshold: float,
        title: str = "Análisis de Violaciones VaR",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(returns.index, returns.values, linewidth=1, 
                label='Retornos', alpha=0.7)

        ax.axhline(y=var_threshold/100, color='red', linestyle='--', 
                   linewidth=2, label=f'VaR Threshold ({var_threshold:.2f}%)')

        breaches = returns[returns < var_threshold/100]
        if len(breaches) > 0:
            ax.scatter(breaches.index, breaches.values, 
                      color='red', s=50, zorder=5, label='Violaciones')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Retorno', fontsize=12)
        ax.set_xlabel('Fecha', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        breach_count = len(breaches)
        breach_rate = (breach_count / len(returns)) * 100
        ax.text(0.02, 0.98, f'Violaciones: {breach_count} ({breach_rate:.2f}%)',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax