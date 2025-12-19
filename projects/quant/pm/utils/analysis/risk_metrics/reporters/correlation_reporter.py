import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.correlation_analyzer import CorrelationAnalyzer


class CorrelationReporter:

    def __init__(self, correlation_analyzer: CorrelationAnalyzer):
        self.analyzer = correlation_analyzer
    
    def generate_report(self, returns: pd.DataFrame) -> None:
        results = self.analyzer.analyze(returns)
        self.print_correlation_summary(results)   

    def print_correlation_summary(self, results: Dict) -> None:

        print("ANÁLISIS DE CORRELACIÓN".center(60))
    
        corr_matrix = results['correlation_matrix']  

        print("MATRIZ DE CORRELACIÓN")
        print(corr_matrix.round(3))
        
        print("\nESTADÍSTICAS DE CORRELACIÓN")
        print(f"  Correlación promedio:    {results['mean_correlation']:>8.3f}")
        print(f"  Correlación máxima:      {results['max_correlation']:>8.3f}")
        print(f"  Correlación mínima:      {results['min_correlation']:>8.3f}")
        print(f"  Desviación estándar:     {results['std_correlation']:>8.3f}")
        
        corr_flat = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack()
        
        print("PARES MÁS CORRELACIONADOS")
        top_pairs = corr_flat.nlargest(3)
        for (asset1, asset2), corr in top_pairs.items():
            print(f"  {asset1} - {asset2}:  {corr:>6.3f}")
        
        print("PARES MENOS CORRELACIONADOS")
        bottom_pairs = corr_flat.nsmallest(3)
        for (asset1, asset2), corr in bottom_pairs.items():
            print(f"  {asset1} - {asset2}:  {corr:>6.3f}")
