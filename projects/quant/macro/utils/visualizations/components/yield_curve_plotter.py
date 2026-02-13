import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional


class YieldCurvePlotter:

    COLORS = {
        'bg':       '#ffffff',
        'spot':     '#0984e3',
        'forward':  '#e74c3c',
        'text':     '#2c3e50',
        'muted':    '#7f8c8d',
        'grid':     '#dfe6e9',
        'border':   '#dee2e6',
        'fill_up':  '#ffe0dc',
        'fill_down':'#d4edda',
    }

    def plot_yield_curve(
        self,
        rates: Dict[str, float],
        figsize: tuple = (10, 6),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor='#f0f2f5')
            is_own_figure = True
        else:
            is_own_figure = False

        ax.set_facecolor(self.COLORS['bg'])

        tenor_order = ['2Y', '5Y', '10Y', '30Y']
        tenors = [t for t in tenor_order if t in rates]
        x = list(range(len(tenors)))
        values = [rates[t] for t in tenors]

        ax.plot(
            x, values, marker='o', linewidth=2.5, markersize=10,
            color=self.COLORS['spot'], label='Curva Actual', zorder=5,
            markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor=self.COLORS['spot'],
        )
        ax.fill_between(x, values, alpha=0.06, color=self.COLORS['spot'])

        for xi, v in zip(x, values):
            ax.annotate(
                f'{v:.2f}%', (xi, v),
                textcoords='offset points', xytext=(0, 14),
                ha='center', fontsize=9.5, fontweight='bold',
                color=self.COLORS['spot'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.85),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(tenors, fontsize=11, fontweight='bold')
        ax.set_xlabel('Tenor', fontsize=10, color=self.COLORS['muted'])
        ax.set_ylabel('Tasa de Interés (%)', fontsize=10, color=self.COLORS['muted'])
        ax.set_title('Curva de Tipos de Interés (USA)', fontsize=14,
                     fontweight='bold', color=self.COLORS['text'])
        ax.grid(True, alpha=0.15, color=self.COLORS['grid'], linestyle='--')
        ax.legend(fontsize=9, framealpha=0.95, edgecolor=self.COLORS['border'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(self.COLORS['border'])

        if is_own_figure:
            try:
                plt.tight_layout()
            except Exception:
                pass

        return ax

    def plot_yield_curve_with_forwards(
        self,
        spot_rates: Dict[str, float],
        forward_rates: Dict[str, float],
        figsize: tuple = (12, 7),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, facecolor='#f0f2f5')

        ax.set_facecolor(self.COLORS['bg'])

        tenor_order = ['3M', '2Y', '5Y', '10Y', '30Y']
        tenors = [t for t in tenor_order if t in spot_rates]
        x = list(range(len(tenors)))
        values = [spot_rates[t] for t in tenors]

        # spot curve
        ax.plot(
            x, values, marker='o', linewidth=2.5, markersize=10,
            color=self.COLORS['spot'], label='Spot (Actual)', zorder=5,
            markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor=self.COLORS['spot'],
        )
        ax.fill_between(x, values, alpha=0.06, color=self.COLORS['spot'])

        for xi, v in zip(x, values):
            ax.annotate(
                f'{v:.2f}%', (xi, v),
                textcoords='offset points', xytext=(0, 14),
                ha='center', fontsize=9, fontweight='bold',
                color=self.COLORS['spot'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.85),
            )

        # forward curve
        fwd_map: Dict[str, float] = {}
        for key, rate in forward_rates.items():
            parts = key.split('→')
            if len(parts) == 2 and parts[1] in tenor_order:
                fwd_map.setdefault(parts[1], rate)

        fwd_x, fwd_y = [], []
        for i, t in enumerate(tenors):
            if t in fwd_map:
                fwd_x.append(i)
                fwd_y.append(fwd_map[t])

        if fwd_x:
            ax.plot(
                fwd_x, fwd_y, marker='D', linewidth=2, markersize=8,
                color=self.COLORS['forward'], linestyle='--',
                label='Forward Implícito', zorder=4, alpha=0.9,
                markerfacecolor='white', markeredgewidth=2,
                markeredgecolor=self.COLORS['forward'],
            )
            for xi_f, fv in zip(fwd_x, fwd_y):
                ax.annotate(
                    f'{fv:.2f}%', (xi_f, fv),
                    textcoords='offset points', xytext=(0, -18),
                    ha='center', fontsize=8.5, color=self.COLORS['forward'],
                    bbox=dict(boxstyle='round,pad=0.2', fc='white',
                              ec='none', alpha=0.85),
                )

            # shade gap
            for i, t in enumerate(tenors):
                if t in fwd_map:
                    fv = fwd_map[t]
                    sv = values[i]
                    clr = self.COLORS['fill_up'] if fv > sv else self.COLORS['fill_down']
                    ax.fill_between(
                        [i - 0.18, i + 0.18], [sv, sv], [fv, fv],
                        alpha=0.50, color=clr, zorder=2,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(tenors, fontsize=11, fontweight='bold')
        ax.set_xlabel('Tenor', fontsize=10, color=self.COLORS['muted'])
        ax.set_ylabel('Yield (%)', fontsize=10, color=self.COLORS['muted'])
        ax.set_title('Curva de Tipos: Spot vs Forward Implícito',
                     fontsize=14, fontweight='bold', color=self.COLORS['text'])
        ax.grid(True, alpha=0.15, color=self.COLORS['grid'], linestyle='--')
        ax.legend(fontsize=10, framealpha=0.95, edgecolor=self.COLORS['border'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(self.COLORS['border'])

        return ax
