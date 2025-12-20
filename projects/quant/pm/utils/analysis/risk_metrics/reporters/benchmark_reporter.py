import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.benchmark_analyzer import BenchmarkAnalyzer

class BenchmarkReporter:

    def __init__(self, benchmark_analyzer: BenchmarkAnalyzer):
        self.analyzer = benchmark_analyzer
    
    def generate_report(
        self,
        returns: pd.DataFrame, 
        weights: np.ndarray,   
        benchmark_returns: pd.Series,
        risk_free_rate: float
    ) -> None:

        results = self.analyzer.analyze(
            returns, weights, benchmark_returns, risk_free_rate
        )
        self.print_benchmark_analysis(results)
    def print_benchmark_analysis(self, results: Dict) -> None:
        
        print("ANÁLISIS VS BENCHMARK".center(60))
    
        print("TRACKING ERROR")
        print(f"  Tracking Error (diario):  {results['tracking_error_daily']*100:>8.2f}%")
        print(f"  Tracking Error (anual):   {results['tracking_error_annual']*100:>8.2f}%")
        self._interpret_te(results['tracking_error_annual']) 
        
        print("INFORMATION RATIO")
        print(f"  Information Ratio:        {results['information_ratio']:>8.3f}")
        self._interpret_ir(results['information_ratio'])
        
        print("BETA")
        print(f"  Beta:                     {results['beta']:>8.3f}")
        print(f"  R²:                       {results['r_squared']:>8.3f}")
        print(f"  Correlación:              {results['correlation']:>8.3f}")
        self._interpret_beta(results['beta'])
        
        print("ALPHA (Jensen)")
        print(f"  Alpha (anualizado):       {results['alpha_annual']*100:>8.2f}%")  
        print(f"  Retorno cartera:          {results['portfolio_return_annual']*100:>8.2f}%")
        print(f"  Retorno benchmark:        {results['benchmark_return_annual']*100:>8.2f}%")
        print(f"  Retorno esperado (CAPM):  {results['expected_return']*100:>8.2f}%")
        self._interpret_alpha(results['alpha_annual'])  

    def _interpret_te(self, te: float) -> None:

        te_pct = te * 100
        if te_pct < 2:
            print(f"  Interpretación:          Muy cercano al benchmark")
        elif te_pct < 5:
            print(f"  Interpretación:          Desviación moderada")
        elif te_pct < 10:
            print(f"  Interpretación:          Gestión activa notable")
        else:
            print(f"  Interpretación:          Alta desviación del benchmark")
    
    def _interpret_ir(self, ir: float) -> None:

        if ir > 0.5:
            print(f"  Interpretación:          Excelente - supera al benchmark")
        elif ir > 0:
            print(f"  Interpretación:          Positivo - añade valor")
        elif ir > -0.5:
            print(f"  Interpretación:          Ligeramente inferior")
        else:
            print(f"  Interpretación:          Bajo desempeño significativo")
    
    def _interpret_beta(self, beta: float) -> None:

        if beta > 1.2:
            print(f"  Interpretación:          Alta sensibilidad (agresivo)")
        elif beta > 0.8:
            print(f"  Interpretación:          Similar al mercado")
        elif beta > 0:
            print(f"  Interpretación:          Baja sensibilidad (defensivo)")
        else:
            print(f"  Interpretación:          Correlación inversa")
    
    def _interpret_alpha(self, alpha: float) -> None:

        alpha_pct = alpha * 100
        if alpha_pct > 5:
            print(f"  Interpretación:          Excelente - supera expectativas")
        elif alpha_pct > 0:
            print(f"  Interpretación:          Positivo - genera valor")
        elif alpha_pct > -5:
            print(f"  Interpretación:          Ligeramente por debajo")
        else:
            print(f"  Interpretación:          Bajo desempeño notable")