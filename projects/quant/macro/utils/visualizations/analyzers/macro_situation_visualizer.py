import matplotlib.pyplot as plt
from typing import Dict, Optional
from ...analyzers.macro_situation_analyzer import MacroSituationAnalyzer
from ..components.yield_curve_plotter import YieldCurvePlotter

class MacroSituationVisualizer:
    
    def __init__(self, analyzer: Optional[MacroSituationAnalyzer] = None):
        self.analyzer = analyzer if analyzer is not None else MacroSituationAnalyzer()
        self.yield_curve_plotter = YieldCurvePlotter()
    
    def plot_macro_situation(
        self,
        situation_analysis: Dict,
        figsize: tuple = (16, 12)
    ) -> plt.Figure:
        """
        Genera dashboard completo de situación macroeconómica.
        
        Args:
            situation_analysis: Dict con resultados de MacroSituationAnalyzer
            figsize: Tamaño de la figura
            
        Returns:
            Figure de matplotlib con el dashboard
        """
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        yield_curve = situation_analysis.get('yield_curve', {})
        # Soporte para dataclass o dict
        levels = getattr(yield_curve, 'levels', yield_curve.get('levels', {})) if hasattr(yield_curve, 'get') else getattr(yield_curve, 'levels', {})
        if levels:
            self.yield_curve_plotter.plot_yield_curve(levels, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        inflation = situation_analysis.get('inflation', {})
        if inflation:
            commodities = []
            changes = []
            # Soporte para dataclass o dict
            commodity_changes = getattr(inflation, 'commodity_changes', inflation.get('commodity_changes', {})) if hasattr(inflation, 'get') else getattr(inflation, 'commodity_changes', {})
            commodity_names = getattr(inflation, 'commodity_names', inflation.get('commodity_names', {})) if hasattr(inflation, 'get') else getattr(inflation, 'commodity_names', {})
            
            for key, change_value in commodity_changes.items():
                if isinstance(change_value, (int, float)):
                    name = commodity_names.get(key, key.upper())
                    commodities.append(name)
                    changes.append(change_value)
            
            if commodities:
                colors = ['red' if c > 10 else 'green' if c > 0 else 'gray' for c in changes]
                bars = ax2.barh(commodities, changes, color=colors, alpha=0.7, edgecolor='black')
                ax2.axvline(0, color='black', linestyle='-', linewidth=1)
                ax2.set_xlabel('Cambio (%)', fontsize=11, fontweight='bold')
                ax2.set_title('Señales de Inflación (12 meses)', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')

                for bar, val in zip(bars, changes):
                    width = bar.get_width()
                    ax2.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}%', ha='left' if width > 0 else 'right', 
                           va='center', fontsize=9)

        ax3 = fig.add_subplot(gs[1, 0])
        credit = situation_analysis.get('credit', {})
        # Soporte para dataclass o dict
        vix_value = getattr(credit, 'vix_level', credit.get('vix_level')) if hasattr(credit, 'get') else getattr(credit, 'vix_level', None)
        if vix_value is not None: 
            ax3.barh(['VIX'], [vix_value], color='green' if vix_value < 20 else 'orange' if vix_value < 25 else 'red',
                   alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Nivel VIX', fontsize=11, fontweight='bold')
            ax3.set_title('Volatilidad del Mercado', fontsize=12, fontweight='bold')
            ax3.text(vix_value + 1, 0, f'{vix_value:.2f}', va='center', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')

        ax4 = fig.add_subplot(gs[1, 1])
        bonds = situation_analysis.get('global_bonds', {})
        if bonds:
            regions = []
            changes_1y = []
            for region, data in bonds.items():
                if isinstance(data, dict) and 'change_1y' in data:
                    regions.append(region)
                    changes_1y.append(data['change_1y'])
            
            if regions:
                colors = ['green' if c > 5 else 'red' if c < -5 else 'gray' for c in changes_1y]
                bars = ax4.barh(regions, changes_1y, color=colors, alpha=0.7, edgecolor='black')
                ax4.axvline(0, color='black', linestyle='-', linewidth=1)
                ax4.set_xlabel('Cambio 1 Año (%)', fontsize=11, fontweight='bold')
                ax4.set_title('Bonos Soberanos Globales', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')
                
                for bar, val in zip(bars, changes_1y):
                    width = bar.get_width()
                    ax4.text(width + (0.5 if width > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                           f'{val:.1f}%', ha='left' if width > 0 else 'right', 
                           va='center', fontsize=9)

        ax5 = fig.add_subplot(gs[2, :])
        sentiment = situation_analysis.get('risk_sentiment', {})
        if sentiment:
            ax5.axis('off')
            # Soporte para dataclass o dict
            fear_level = getattr(sentiment, 'fear_level', sentiment.get('fear_level', 'N/A')) if hasattr(sentiment, 'get') else getattr(sentiment, 'fear_level', 'N/A')
            dollar_strength = getattr(sentiment, 'dollar_strength', sentiment.get('dollar_strength', 'N/A')) if hasattr(sentiment, 'get') else getattr(sentiment, 'dollar_strength', 'N/A')
            safe_haven = getattr(sentiment, 'safe_haven', sentiment.get('safe_haven', 'N/A')) if hasattr(sentiment, 'get') else getattr(sentiment, 'safe_haven', 'N/A')
            
            sentiment_text = f"""
            SENTIMIENTO DE RIESGO
            
            Nivel de miedo:           {fear_level}
            Fortaleza del dólar:      {dollar_strength}
            Demanda de refugio:       {safe_haven}
            """
            ax5.text(0.1, 0.5, sentiment_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=ax5.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Situación Macroeconómica Global - Dashboard',
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig