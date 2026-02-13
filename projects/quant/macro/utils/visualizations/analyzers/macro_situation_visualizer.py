import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Dict, Optional
from ...analyzers.macro_situation_analyzer import MacroSituationAnalyzer
from ..components.yield_curve_plotter import YieldCurvePlotter


class MacroSituationVisualizer:

    COLORS = {
        'bg':           '#f0f2f5',
        'panel':        '#ffffff',
        'title':        '#1a1a2e',
        'subtitle':     '#34495e',
        'text':         '#2c3e50',
        'text_muted':   '#7f8c8d',
        'positive':     '#00b894',
        'negative':     '#e17055',
        'warning':      '#fdcb6e',
        'neutral':      '#b2bec3',
        'blue':         '#0984e3',
        'blue_light':   '#74b9ff',
        'orange':       '#e17055',
        'purple':       '#6c5ce7',
        'grid':         '#dfe6e9',
        'border':       '#dee2e6',
        'vix_low':      '#00b894',
        'vix_moderate': '#00cec9',
        'vix_elevated': '#fdcb6e',
        'vix_high':     '#e17055',
        'vix_extreme':  '#d63031',
    }

    def __init__(self, analyzer: Optional[MacroSituationAnalyzer] = None):
        self.analyzer = analyzer if analyzer is not None else MacroSituationAnalyzer()
        self.yield_curve_plotter = YieldCurvePlotter()

    # ─── helpers ────────────────────────────────────────────────────
    def _get_attr(self, obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _style_panel(self, ax, title='', icon=''):
        ax.set_facecolor(self.COLORS['panel'])
        for spine in ax.spines.values():
            spine.set_color(self.COLORS['border'])
            spine.set_linewidth(0.5)
        if title:
            label = f'{icon}  {title}' if icon else title
            ax.set_title(label, fontsize=13, fontweight='bold',
                         color=self.COLORS['title'], pad=14, loc='left')

    # ─── main entry point ──────────────────────────────────────────
    def plot_macro_situation(
        self,
        situation_analysis: Dict,
        figsize: tuple = (20, 14)
    ) -> plt.Figure:

        fig = plt.figure(figsize=figsize, facecolor=self.COLORS['bg'])
        gs = fig.add_gridspec(
            3, 4,
            hspace=0.40, wspace=0.35,
            left=0.06, right=0.94, top=0.87, bottom=0.03,
        )

        # ── title ──
        fig.suptitle(
            'GLOBAL MACROECONOMIC SITUATION',
            fontsize=22, fontweight='bold',
            color=self.COLORS['title'], y=0.97,
        )

        # ── risk badge ──
        summary = situation_analysis.get('summary', {})
        overall = summary.get('overall_risk', '') if isinstance(summary, dict) else ''
        score   = summary.get('risk_score', 0)   if isinstance(summary, dict) else 0

        if overall:
            badge_color = (
                self.COLORS['positive'] if 'LOW' in str(overall).upper()
                else self.COLORS['warning'] if 'MODERATE' in str(overall).upper()
                else self.COLORS['negative'] if overall
                else self.COLORS['neutral']
            )
            fig.text(
                0.5, 0.925,
                f'  Risk: {overall}  │  Score: {score}  ',
                fontsize=12, ha='center', va='center', fontweight='bold',
                color=badge_color,
                bbox=dict(
                    boxstyle='round,pad=0.5', facecolor=badge_color,
                    alpha=0.12, edgecolor=badge_color, linewidth=1.5,
                ),
            )

        # ── panels ──
        ax_yield = fig.add_subplot(gs[0, :2])
        self._plot_yield_curve(ax_yield, situation_analysis)

        ax_infl = fig.add_subplot(gs[0, 2:])
        self._plot_inflation(ax_infl, situation_analysis)

        ax_vix = fig.add_subplot(gs[1, :2])
        self._plot_vix_gauge(ax_vix, situation_analysis)

        ax_bonds = fig.add_subplot(gs[1, 2:])
        self._plot_global_bonds(ax_bonds, situation_analysis)

        ax_sent = fig.add_subplot(gs[2, :])
        self._plot_sentiment(ax_sent, situation_analysis)

        return fig

    # ─── Panel 1: Yield Curve ──────────────────────────────────────
    def _plot_yield_curve(self, ax, data):
        self._style_panel(ax, 'US Yield Curve', '▲')

        yield_curve = data.get('yield_curve', {})
        levels = self._get_attr(yield_curve, 'levels', {})
        implied = data.get('implied_yield_curve')

        tenor_order = ['2Y', '5Y', '10Y', '30Y']
        tenors = [t for t in tenor_order if t in levels]

        if not tenors:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12,
                    color=self.COLORS['text_muted'])
            return

        x = list(range(len(tenors)))
        values = [levels[t] for t in tenors]

        # spot curve
        ax.plot(
            x, values, marker='o', linewidth=2.5, markersize=10,
            color=self.COLORS['blue'], label='Spot (Actual)', zorder=5,
            markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor=self.COLORS['blue'],
        )
        ax.fill_between(x, values, alpha=0.06, color=self.COLORS['blue'])

        for xi, v in zip(x, values):
            ax.annotate(
                f'{v:.2f}%', (xi, v),
                textcoords='offset points', xytext=(0, 14),
                ha='center', fontsize=9.5, fontweight='bold',
                color=self.COLORS['blue'],
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.85),
            )

        # forward overlay
        if implied:
            forwards = self._get_attr(implied, 'forward_rates', {})
            if forwards:
                fwd_map = {}
                for key, rate in forwards.items():
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
                        color='#e74c3c', linestyle='--',
                        label='Implied Forward', zorder=4, alpha=0.9,
                        markerfacecolor='white', markeredgewidth=2,
                        markeredgecolor='#e74c3c',
                    )
                    for xi_f, fv in zip(fwd_x, fwd_y):
                        ax.annotate(
                            f'{fv:.2f}%', (xi_f, fv),
                            textcoords='offset points', xytext=(0, -18),
                            ha='center', fontsize=8.5, color='#e74c3c',
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec='none', alpha=0.85),
                        )
                    # shade spot ↔ forward
                    for i, t in enumerate(tenors):
                        if t in fwd_map:
                            fv = fwd_map[t]
                            sv = values[i]
                            clr = '#ffe0dc' if fv > sv else '#d4edda'
                            ax.fill_between(
                                [i - 0.18, i + 0.18],
                                [sv, sv], [fv, fv],
                                alpha=0.50, color=clr, zorder=2,
                            )

        ax.set_xticks(x)
        ax.set_xticklabels(tenors, fontsize=11, fontweight='bold')
        ax.set_ylabel('Yield (%)', fontsize=10, color=self.COLORS['text_muted'])
        ax.grid(True, alpha=0.15, color=self.COLORS['grid'], linestyle='--')
        ax.legend(fontsize=9, loc='upper left', framealpha=0.95,
                  edgecolor=self.COLORS['border'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ─── Panel 2: Inflation (lollipop) ─────────────────────────────
    def _plot_inflation(self, ax, data):
        self._style_panel(ax, 'Inflation Signals (12 months)', '■')

        inflation = data.get('inflation', {})
        if not inflation:
            return

        changes = self._get_attr(inflation, 'commodity_changes', {})
        names   = self._get_attr(inflation, 'commodity_names', {})

        items = []
        for key, val in changes.items():
            if isinstance(val, (int, float)):
                name = names.get(key, key.upper())
                items.append((name, val))
        if not items:
            return

        items.sort(key=lambda x: x[1])
        labels, vals = zip(*items)
        y = list(range(len(labels)))
        max_abs = max(abs(v) for v in vals) if vals else 1
        offset = max(max_abs * 0.08, 1.5)

        for i, (name, val) in enumerate(zip(labels, vals)):
            color = (self.COLORS['negative'] if val > 10
                     else self.COLORS['positive'] if val >= 0
                     else self.COLORS['neutral'])
            ax.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=3, alpha=0.7)
            ax.plot(val, i, 'o', color=color, markersize=12, zorder=5,
                    markeredgecolor='white', markeredgewidth=2)
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + (offset if val >= 0 else -offset), i,
                    f'{val:.1f}%', va='center', ha=ha,
                    fontsize=10, fontweight='bold', color=color)

        ax.axvline(0, color=self.COLORS['border'], linewidth=1, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11, fontweight='bold',
                           color=self.COLORS['text'])
        ax.grid(True, alpha=0.1, axis='x', color=self.COLORS['grid'],
                linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    # ─── Panel 3: VIX Gauge ───────────────────────────────────────
    def _plot_vix_gauge(self, ax, data):
        ax.set_facecolor(self.COLORS['panel'])
        for spine in ax.spines.values():
            spine.set_color(self.COLORS['border'])
            spine.set_linewidth(0.5)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-0.7, 1.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        ax.text(0, 1.38, 'Volatilidad del Mercado (VIX)',
                ha='center', fontsize=13, fontweight='bold',
                color=self.COLORS['title'])

        credit = data.get('credit', {})
        vix = self._get_attr(credit, 'vix_level')
        if vix is None:
            ax.text(0, 0, 'Sin datos', ha='center', fontsize=12,
                    color=self.COLORS['text_muted'])
            return

        outer_r = 1.1
        width = 0.30
        n = 150
        angles = np.linspace(180, 0, n + 1)

        color_stops = [
            (0,  self.COLORS['vix_low']),
            (15, self.COLORS['vix_moderate']),
            (20, self.COLORS['vix_elevated']),
            (30, self.COLORS['vix_high']),
            (40, self.COLORS['vix_extreme']),
        ]

        def _vix_color(v):
            c = self.COLORS['vix_low']
            for thr, col in color_stops:
                if v >= thr:
                    c = col
            return c

        # background arc (dimmed)
        for i in range(n):
            vix_at = (i / n) * 50
            wedge = mpatches.Wedge(
                (0, 0), outer_r, angles[i + 1], angles[i],
                width=width, facecolor=_vix_color(vix_at),
                alpha=0.20, edgecolor='none',
            )
            ax.add_patch(wedge)

        # highlighted arc up to current VIX
        vix_capped = min(max(vix, 0), 50)
        n_hl = max(int(n * (vix_capped / 50)), 2)
        hl_end = 180 - (vix_capped / 50) * 180
        hl_angles = np.linspace(180, hl_end, n_hl + 1)

        for i in range(n_hl):
            vix_at = (i / n_hl) * vix_capped
            wedge = mpatches.Wedge(
                (0, 0), outer_r, hl_angles[i + 1], hl_angles[i],
                width=width, facecolor=_vix_color(vix_at),
                alpha=0.85, edgecolor='none',
            )
            ax.add_patch(wedge)

        # needle
        needle_angle = np.radians(180 - (vix_capped / 50) * 180)
        needle_len = outer_r - width - 0.08
        nx = needle_len * np.cos(needle_angle)
        ny = needle_len * np.sin(needle_angle)
        ax.annotate(
            '', xy=(nx, ny), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=self.COLORS['title'],
                            lw=2.5, mutation_scale=15),
        )
        ax.plot(0, 0, 'o', color=self.COLORS['title'], markersize=8, zorder=11)

        # scale labels
        for v_label in [0, 15, 25, 40, 50]:
            a = np.radians(180 - (v_label / 50) * 180)
            lx = (outer_r + 0.18) * np.cos(a)
            ly = (outer_r + 0.18) * np.sin(a)
            ax.text(lx, ly, str(v_label), ha='center', va='center',
                    fontsize=8.5, color=self.COLORS['text_muted'])

        # value + label
        vc = _vix_color(vix)
        if vix < 15:
            vl = 'BAJA VOLATILIDAD'
        elif vix < 20:
            vl = 'VOL. MODERADA'
        elif vix < 30:
            vl = 'VOL. ELEVADA'
        elif vix < 40:
            vl = 'VOL. ALTA'
        else:
            vl = 'VOL. EXTREMA'

        ax.text(0, -0.22, f'{vix:.1f}', ha='center', fontsize=34,
                fontweight='bold', color=vc)
        ax.text(0, -0.52, vl, ha='center', fontsize=11, fontweight='bold',
                color=vc,
                bbox=dict(boxstyle='round,pad=0.3', fc=vc,
                          alpha=0.10, ec='none'))

    # ─── Panel 4: Global Bonds (lollipop) ─────────────────────────
    def _plot_global_bonds(self, ax, data):
        self._style_panel(ax, 'Bonos Soberanos Globales', '◆')

        bonds = data.get('global_bonds', {})
        if not bonds:
            return

        regions, changes = [], []
        for region, bdata in bonds.items():
            if isinstance(bdata, dict) and 'change_1y' in bdata:
                regions.append(region)
                changes.append(bdata['change_1y'])
        if not regions:
            return

        sorted_data = sorted(zip(regions, changes), key=lambda x: x[1])
        regions, changes = zip(*sorted_data)
        y = list(range(len(regions)))
        max_abs = max(abs(v) for v in changes) if changes else 1
        offset = max(max_abs * 0.06, 0.8)

        for i, (name, val) in enumerate(zip(regions, changes)):
            color = (self.COLORS['positive'] if val > 5
                     else self.COLORS['negative'] if val < -5
                     else self.COLORS['neutral'])
            ax.hlines(y=i, xmin=0, xmax=val, color=color, linewidth=3, alpha=0.7)
            ax.plot(val, i, 'o', color=color, markersize=11, zorder=5,
                    markeredgecolor='white', markeredgewidth=2)
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + (offset if val >= 0 else -offset), i,
                    f'{val:.1f}%', va='center', ha=ha,
                    fontsize=9, fontweight='bold', color=color)

        ax.axvline(0, color=self.COLORS['border'], linewidth=1, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(regions, fontsize=10, fontweight='bold',
                           color=self.COLORS['text'])
        ax.set_xlabel('1 Year Change (%)', fontsize=10,
                      color=self.COLORS['text_muted'])
        ax.grid(True, alpha=0.1, axis='x', color=self.COLORS['grid'],
                linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)

    # ─── Panel 5: Sentiment cards ─────────────────────────────────
    def _plot_sentiment(self, ax, data):
        ax.set_facecolor(self.COLORS['bg'])
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

        sentiment = data.get('risk_sentiment', {})
        summary   = data.get('summary', {})

        fear   = self._get_attr(sentiment, 'fear_level', 'N/A')
        dollar = self._get_attr(sentiment, 'dollar_strength', 'N/A')
        haven  = self._get_attr(sentiment, 'safe_haven', 'N/A')

        risk_factors = (
            summary.get('risk_factors', []) if isinstance(summary, dict)
            else getattr(summary, 'risk_factors', [])
        )

        ax.text(0.5, 0.96, 'SENTIMIENTO DE RIESGO',
                transform=ax.transAxes, ha='center', fontsize=14,
                fontweight='bold', color=self.COLORS['title'])

        cards = [
            ('FEAR',  'Fear Level',           str(fear),   self._color_for_fear(fear)),
            ('USD',   'Dollar Strength',      str(dollar), self._color_for_dollar(dollar)),
            ('HAVEN', 'Safe-Haven Demand',    str(haven),  self._color_for_haven(haven)),
        ]

        card_w = 0.27
        gap    = 0.04
        total  = len(cards) * card_w + (len(cards) - 1) * gap
        x0     = 0.5 - total / 2
        cy, ch = 0.22, 0.62

        for i, (icon, title, value, color) in enumerate(cards):
            cx = x0 + i * (card_w + gap)

            # card bg
            card = FancyBboxPatch(
                (cx, cy), card_w, ch,
                boxstyle='round,pad=0.015',
                facecolor='white', edgecolor=color,
                linewidth=2.5, alpha=0.95,
                transform=ax.transAxes, zorder=3,
            )
            ax.add_patch(card)

            # accent bar
            accent = FancyBboxPatch(
                (cx + 0.02, cy + ch - 0.06), card_w - 0.04, 0.04,
                boxstyle='round,pad=0.005',
                facecolor=color, edgecolor='none', alpha=0.8,
                transform=ax.transAxes, zorder=4,
            )
            ax.add_patch(accent)

            # icon (text label)
            ax.text(cx + card_w / 2, cy + ch - 0.17, icon,
                    transform=ax.transAxes, ha='center', fontsize=14,
                    fontweight='bold', color=color, zorder=5)
            # label
            ax.text(cx + card_w / 2, cy + ch - 0.28, title,
                    transform=ax.transAxes, ha='center', fontsize=10,
                    color=self.COLORS['text_muted'], fontweight='bold', zorder=5)
            # value
            fs = 11 if len(value) < 20 else 9.5 if len(value) < 30 else 8
            ax.text(cx + card_w / 2, cy + 0.18, value,
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=fs, color=color, fontweight='bold', zorder=5,
                    linespacing=1.3)

        # risk factors
        if risk_factors:
            factors_text = ' │ '.join(risk_factors[:4])
            if len(risk_factors) > 4:
                factors_text += f' (+{len(risk_factors) - 4} more)'
            ax.text(
                0.5, 0.10, f'►  {factors_text}',
                transform=ax.transAxes, ha='center', fontsize=8.5,
                color=self.COLORS['text_muted'], fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.3', fc='white',
                          ec=self.COLORS['border'], alpha=0.8),
            )

    # ─── color helpers ────────────────────────────────────────────
    def _color_for_fear(self, fear):
        s = str(fear).upper()
        if 'LOW' in s or 'COMPLACEN' in s:
            return self.COLORS['positive']
        if 'HIGH' in s or 'PANIC' in s or 'EXTREME' in s:
            return self.COLORS['negative']
        return self.COLORS['warning']

    def _color_for_dollar(self, dollar):
        s = str(dollar).upper()
        if 'STRONG' in s:
            return self.COLORS['blue']
        if 'WEAK' in s:
            return self.COLORS['negative']
        return self.COLORS['neutral']

    def _color_for_haven(self, haven):
        s = str(haven).upper()
        if 'NO ' in s:
            return self.COLORS['positive']
        if 'COOLING' in s or 'CORRECTING' in s:
            return self.COLORS['warning']
        if 'HIGH' in s or 'EXTREME' in s:
            return self.COLORS['negative']
        return self.COLORS['neutral']
