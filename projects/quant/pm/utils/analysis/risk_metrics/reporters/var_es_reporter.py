import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ..analyzers.var_es_analyzer import VarEsAnalyzer
from ....tools.config import (
    RISK_ANALYSIS,
    MONTE_CARLO_SIMULATIONS,
    MONTE_CARLO_SEED,
    VAR_RISK_LEVELS,
)

class VarEsReporter:
    def __init__(self, var_es_analyzer: VarEsAnalyzer):
        self.analyzer = var_es_analyzer

    def generate_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = None,
        n_simulations: int = None,
        seed: int = None,
    ) -> None:
        confidence_level = confidence_level if confidence_level is not None else RISK_ANALYSIS['default_confidence_level']
        n_simulations = n_simulations if n_simulations is not None else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED

        results = self.analyzer.calculate_multi_level(
            returns=returns,
            weights=weights,
            confidence_levels=(confidence_level,),
            n_simulations=n_simulations,
            seed=seed,
        )

        comparison = self._results_to_dataframe(results[confidence_level])
        self._print_comparison(comparison, confidence_level)

    def generate_multi_level_report(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_levels: Tuple[float, ...] = None,
        method: str = 'historical',
        n_simulations: int = None,
        seed: int = None,
    ) -> None:
        confidence_levels = confidence_levels if confidence_levels is not None else RISK_ANALYSIS['multi_level_confidence']
        n_simulations = n_simulations if n_simulations is not None else MONTE_CARLO_SIMULATIONS
        seed = seed if seed is not None else MONTE_CARLO_SEED

        results = self.analyzer.calculate_multi_level(
            returns=returns,
            weights=weights,
            confidence_levels=confidence_levels,
            methods=[method],
            n_simulations=n_simulations,
            seed=seed,
        )

        self._print_multi_level(results, confidence_levels, method)

    def _results_to_dataframe(self, method_results: Dict) -> pd.DataFrame:
        data = [
            {
                'Method': method.capitalize(),
                'VaR_daily_%': values['var_daily_pct'],
                'VaR_annual_%': values['var_annual_pct'],
                'ES_daily_%': values['es_daily_pct'],
                'ES_annual_%': values['es_annual_pct'],
            }
            for method, values in method_results.items()
        ]
        return pd.DataFrame(data).set_index('Method')

    @staticmethod
    def _classify_risk(abs_var_daily: float) -> str:
        if abs_var_daily < VAR_RISK_LEVELS['low']:
            return "Low"
        if abs_var_daily < VAR_RISK_LEVELS['moderate']:
            return "Moderate"
        if abs_var_daily < VAR_RISK_LEVELS['high']:
            return "High"
        return "Very High"

    def _print_comparison(self, comparison: pd.DataFrame, confidence_level: float) -> None:
        header = f"VaR AND ES ANALYSIS (Confidence level: {confidence_level * 100:.0f}%)"
        print(header.center(70))

        row_fmt = "{:<15} {:>7.2f}%      {:>7.2f}%      {:>7.2f}%      {:>7.2f}%"
        print(f"{'Method':<15} {'Daily VaR':<15} {'Annual VaR':<15} {'Daily ES':<15} {'Annual ES':<15}")
        for method, row in comparison.iterrows():
            print(row_fmt.format(method, row['VaR_daily_%'], row['VaR_annual_%'], row['ES_daily_%'], row['ES_annual_%']))

        avg_var = comparison['VaR_daily_%'].mean()
        avg_es = comparison['ES_daily_%'].mean()

        print("INTERPRETATION")
        print(f"  Average daily VaR:       {avg_var:.2f}%")
        print(f"  Average daily ES:        {avg_es:.2f}%")
        print(f"  Max expected loss:       {avg_es:.2f}% on an adverse day")
        print(f"  Risk level:              {self._classify_risk(abs(avg_var))}")

    def _print_multi_level(self, results: Dict, confidence_levels: Tuple[float, ...], method: str) -> None:
        print(f"VaR and ES - Method: {method.capitalize()}".center(70))
        print(f"{'Confidence':<12} {'Daily VaR':<15} {'Annual VaR':<15} {'Daily ES':<15} {'Annual ES':<15}")

        row_fmt = "{:>5.0f}%       {:>7.2f}%      {:>7.2f}%      {:>7.2f}%      {:>7.2f}%"
        for cl in confidence_levels:
            v = results[cl][method]
            print(row_fmt.format(cl * 100, v['var_daily_pct'], v['var_annual_pct'], v['es_daily_pct'], v['es_annual_pct']))
