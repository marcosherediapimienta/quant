import pandas as pd
from ..analyzers.portfolio_optimization_analyzer import PortfolioOptimizationAnalyzer
from ....tools.config import MIN_WEIGHT_DISPLAY

class PortfolioReporter:
    def __init__(self, portfolio_analyzer: PortfolioOptimizationAnalyzer):
        self.analyzer = portfolio_analyzer
    
    def generate_tangent_report(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float,
        allow_short: bool = False
    ) -> None:

        results = self.analyzer.analyze_efficient_frontier(
            returns, risk_free_rate, allow_short=allow_short
        )
        
        tangent = results['tangent_portfolio']
        if tangent is None:
            print(" Could not calculate tangent portfolio")
            return
        
        print("TANGENT PORTFOLIO (Maximum Sharpe)".center(60))
        print("CHARACTERISTICS")
        print(f"  Expected Return:         {tangent['return']*100:>8.2f}%")
        print(f"  Volatility:              {tangent['volatility']*100:>8.2f}%")
        print(f"  Sharpe Ratio:            {tangent['sharpe_ratio']:>8.3f}")
        
        if tangent['weights'] is not None:
            print("PORTFOLIO COMPOSITION")
            weights_df = pd.DataFrame({
                'Asset': tangent['assets'],
                'Weight': tangent['weights']
            })
            weights_df = weights_df[weights_df['Weight'] > MIN_WEIGHT_DISPLAY]
            weights_df = weights_df.sort_values('Weight', ascending=False)
            
            for _, row in weights_df.iterrows():
                print(f"  {row['Asset']:<10} {row['Weight']*100:>7.2f}%")
        
    def generate_minimum_variance_report(
        self,
        returns: pd.DataFrame,
        allow_short: bool = False
    ) -> None:

        results = self.analyzer.analyze_minimum_variance(returns, allow_short)
        
        if results['weights'] is None or len(results['weights']) == 0:
            print(" Could not calculate minimum variance portfolio")
            return

        print("MINIMUM VARIANCE PORTFOLIO".center(60))
        print("CHARACTERISTICS")
        print(f"  Expected Return:         {results['return']*100:>8.2f}%")
        print(f"  Volatility:              {results['volatility']*100:>8.2f}%")
        
        print("PORTFOLIO COMPOSITION")
        weights_df = pd.DataFrame({
            'Asset': results['assets'],
            'Weight': results['weights']
        })
        weights_df = weights_df[weights_df['Weight'] > MIN_WEIGHT_DISPLAY]
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        for _, row in weights_df.iterrows():
            print(f"  {row['Asset']:<10} {row['Weight']*100:>7.2f}%")