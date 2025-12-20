import matplotlib.pyplot as plt
from typing import Dict, Optional

class YieldCurvePlotter:
    
    def plot_yield_curve(
        self,
        rates: Dict[str, float],
        figsize: tuple = (10, 6),
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            is_own_figure = True
        else:
            is_own_figure = False

        tenor_order = ['2Y', '5Y', '10Y', '30Y']
        tenors = [t for t in tenor_order if t in rates]
        values = [rates[t] for t in tenors]
        
        ax.plot(tenors, values, marker='o', linewidth=2, markersize=8, 
            color='blue', label='Curva Actual')
        ax.set_xlabel('Tenor', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tasa de Interés (%)', fontsize=11, fontweight='bold')
        ax.set_title('Curva de Tipos de Interés (USA)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if is_own_figure:
            try:
                plt.tight_layout()
            except:
                pass
        
        return ax