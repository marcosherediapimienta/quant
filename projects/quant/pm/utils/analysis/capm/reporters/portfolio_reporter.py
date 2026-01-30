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
            print("⚠️  No se pudo calcular el portafolio tangente")
            return
        
        print("PORTAFOLIO TANGENTE (Máximo Sharpe)".center(60))
        print("CARACTERÍSTICAS")
        print(f"  Retorno Esperado:        {tangent['return']*100:>8.2f}%")
        print(f"  Volatilidad:             {tangent['volatility']*100:>8.2f}%")
        print(f"  Sharpe Ratio:            {tangent['sharpe_ratio']:>8.3f}")
        
        if tangent['weights'] is not None:
            print("COMPOSICIÓN DEL PORTAFOLIO")
            weights_df = pd.DataFrame({
                'Activo': tangent['assets'],
                'Peso': tangent['weights']
            })
            weights_df = weights_df[weights_df['Peso'] > MIN_WEIGHT_DISPLAY]
            weights_df = weights_df.sort_values('Peso', ascending=False)
            
            for _, row in weights_df.iterrows():
                print(f"  {row['Activo']:<10} {row['Peso']*100:>7.2f}%")
        
    def generate_minimum_variance_report(
        self,
        returns: pd.DataFrame,
        allow_short: bool = False
    ) -> None:

        results = self.analyzer.analyze_minimum_variance(returns, allow_short)
        
        if results['weights'] is None or len(results['weights']) == 0:
            print("⚠️  No se pudo calcular el portafolio de mínima varianza")
            return

        print("PORTAFOLIO DE MÍNIMA VARIANZA".center(60))
        print("CARACTERÍSTICAS")
        print(f"  Retorno Esperado:        {results['return']*100:>8.2f}%")
        print(f"  Volatilidad:             {results['volatility']*100:>8.2f}%")
        
        print("COMPOSICIÓN DEL PORTAFOLIO")
        weights_df = pd.DataFrame({
            'Activo': results['assets'],
            'Peso': results['weights']
        })
        weights_df = weights_df[weights_df['Peso'] > MIN_WEIGHT_DISPLAY]
        weights_df = weights_df.sort_values('Peso', ascending=False)
        
        for _, row in weights_df.iterrows():
            print(f"  {row['Activo']:<10} {row['Peso']*100:>7.2f}%")