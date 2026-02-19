import pandas as pd
import numpy as np
from typing import Dict
from ..analyzers.benchmark_analyzer import BenchmarkAnalyzer
from ....tools.config import (
    TRACKING_ERROR_THRESHOLDS,
    INFORMATION_RATIO_THRESHOLDS,
    BETA_THRESHOLDS,
    ALPHA_THRESHOLDS,
)

class BenchmarkReporter:
    def __init__(self, benchmark_analyzer: BenchmarkAnalyzer):
        self.analyzer = benchmark_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: pd.Series,
        risk_free_rate: float,
    ) -> None:
        results = self.analyzer.analyze(returns, weights, benchmark_returns, risk_free_rate)
        self._print_analysis(results)

    @staticmethod
    def _classify_te(te_annual: float) -> str:
        te_pct = te_annual * 100
        if te_pct < TRACKING_ERROR_THRESHOLDS['very_close']:
            return "Muy cercano al benchmark"
        if te_pct < TRACKING_ERROR_THRESHOLDS['moderate']:
            return "Desviación moderada"
        if te_pct < TRACKING_ERROR_THRESHOLDS['active']:
            return "Gestión activa notable"
        return "Alta desviación del benchmark"

    @staticmethod
    def _classify_ir(ir: float) -> str:
        if ir > INFORMATION_RATIO_THRESHOLDS['excellent']:
            return "Excelente - supera al benchmark"
        if ir > INFORMATION_RATIO_THRESHOLDS['positive']:
            return "Positivo - añade valor"
        if ir > INFORMATION_RATIO_THRESHOLDS['slightly_below']:
            return "Ligeramente inferior"
        return "Bajo desempeño significativo"

    @staticmethod
    def _classify_beta(beta: float) -> str:
        if beta > BETA_THRESHOLDS['aggressive']:
            return "Alta sensibilidad (agresivo)"
        if beta > BETA_THRESHOLDS['market']:
            return "Similar al mercado"
        if beta > 0:
            return "Baja sensibilidad (defensivo)"
        return "Correlación inversa"

    @staticmethod
    def _classify_alpha(alpha_annual: float) -> str:
        alpha_pct = alpha_annual * 100
        if alpha_pct > ALPHA_THRESHOLDS['excellent']:
            return "Excelente - supera expectativas"
        if alpha_pct > ALPHA_THRESHOLDS['positive']:
            return "Positivo - genera valor"
        if alpha_pct > ALPHA_THRESHOLDS['slightly_below']:
            return "Ligeramente por debajo"
        return "Bajo desempeño notable"

    def _print_analysis(self, r: Dict) -> None:
        print("ANÁLISIS VS BENCHMARK".center(60))

        print("TRACKING ERROR")
        print(f"  Tracking Error (diario):  {r['tracking_error_daily'] * 100:>8.2f}%")
        print(f"  Tracking Error (anual):   {r['tracking_error_annual'] * 100:>8.2f}%")
        print(f"  Interpretación:           {self._classify_te(r['tracking_error_annual'])}")

        print("INFORMATION RATIO")
        print(f"  Information Ratio:        {r['information_ratio']:>8.3f}")
        print(f"  Interpretación:           {self._classify_ir(r['information_ratio'])}")

        print("BETA")
        print(f"  Beta:                     {r['beta']:>8.3f}")
        print(f"  R²:                       {r['r_squared']:>8.3f}")
        print(f"  Correlación:              {r['correlation']:>8.3f}")
        print(f"  Interpretación:           {self._classify_beta(r['beta'])}")

        print("ALPHA (Jensen)")
        print(f"  Alpha (anualizado):       {r['alpha_annual'] * 100:>8.2f}%")
        print(f"  Retorno cartera:          {r['portfolio_return_annual'] * 100:>8.2f}%")
        print(f"  Retorno benchmark:        {r['benchmark_return_annual'] * 100:>8.2f}%")
        print(f"  Retorno esperado (CAPM):  {r['expected_return'] * 100:>8.2f}%")
        print(f"  Interpretación:           {self._classify_alpha(r['alpha_annual'])}")
