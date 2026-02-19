import matplotlib.pyplot as plt
import pandas as pd
from ...visualizations.components.frontier_plotter import FrontierPlotter
from ...analysis.capm.analyzers.portfolio_optimization_analyzer import PortfolioOptimizationAnalyzer
from ...tools.config import FRONTIER_POINTS, MIN_WEIGHT_DISPLAY, SML_CONFIG

class PortfolioOptimizationVisualizer:
    def __init__(self, portfolio_analyzer: PortfolioOptimizationAnalyzer):
        self.analyzer = portfolio_analyzer
        self.plotter = FrontierPlotter()
    
    def plot_efficient_frontier_analysis(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float,
        n_points: int = None,
        allow_short: bool = False,
        figsize: tuple = (16, 12)
    ) -> plt.Figure:

        if n_points is None:
            n_points = FRONTIER_POINTS

        results = self.analyzer.analyze_efficient_frontier(
            returns, risk_free_rate, n_points, allow_short
        )
        
        frontier = results['frontier']
        cml = results['cml']
        tangent = results['tangent_portfolio']
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])

        if cml is not None:
            self.plotter.plot_frontier_with_cml(
                frontier, cml, risk_free_rate,
                title="Efficient Frontier and Capital Market Line",
                ax=ax1
            )
        else:
            self.plotter.plot_efficient_frontier(frontier, ax=ax1)

        ax2 = fig.add_subplot(gs[1, 0])

        if tangent and tangent['weights'] is not None:
            weights_df = pd.DataFrame({
                'Asset': tangent['assets'],
                'Weight': tangent['weights']
            })
            weights_df = weights_df[weights_df['Weight'] > MIN_WEIGHT_DISPLAY].sort_values('Weight', ascending=True)
            ax2.barh(weights_df['Asset'], weights_df['Weight'] * 100)
            ax2.set_xlabel('Weight (%)', fontsize=11)
            ax2.set_title('Tangent Portfolio Composition', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'Not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Tangent Portfolio Composition', fontsize=12, fontweight='bold')

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        if tangent:
            stats_text = f"""
            TANGENT PORTFOLIO
            
            Expected Return:   {tangent['return']*100:.2f}%
            Volatility:        {tangent['volatility']*100:.2f}%
            Sharpe Ratio:      {tangent['sharpe_ratio']:.3f}
            
            Risk-Free Rate:    {risk_free_rate*100:.2f}%
            """
        else:
            stats_text = "Not available"
        
        ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax3.transAxes)
        
        plt.suptitle("Portfolio Optimization Analysis", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def plot_sml_analysis(
        self,
        risk_free_rate: float,
        market_return: float,
        asset_betas: dict = None,
        asset_returns: dict = None,
        max_beta: float = None,
        figsize: tuple = (12, 8)
    ) -> plt.Figure:

        if max_beta is None:
            max_beta = SML_CONFIG['max_beta']
   
        sml = self.analyzer.analyze_sml(risk_free_rate, market_return, max_beta)
        
        fig, ax = plt.subplots(figsize=figsize)
        self.plotter.plot_sml(sml, asset_betas, asset_returns, ax=ax)
        
        return fig