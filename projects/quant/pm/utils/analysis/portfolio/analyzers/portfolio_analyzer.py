from typing import List, Dict
from dataclasses import dataclass

from ...valuation.analyzers.company_analyzer import CompanyAnalyzer
from ....data import DataManager  
from ..components.selector import CompanySelector
from ..components.optimizer import WeightOptimizer
from ..components.index_fetcher import IndexFetcher
from ..components.date_utils import DateCalculator
from ..components.returns_calculator import ReturnsCalculator
from ..components.metrics_calculator import PortfolioMetricsCalculator

@dataclass
class PortfolioConfig:
    min_score: float = 60.0
    max_companies: int = 10
    max_per_sector: int = 3
    selection_method: str = 'total_score'
    weight_method: str = 'equal'
    lookback_years: int = 5

class PortfolioAnalyzer:

    def __init__(self, config: PortfolioConfig = None, data_manager: DataManager = None):
        self.config = config if config else PortfolioConfig()
        self.data_manager = data_manager if data_manager else DataManager() 
        self.company_analyzer = CompanyAnalyzer()
        self.selector = CompanySelector(
            min_score=self.config.min_score,
            max_companies=self.config.max_companies,
            max_per_sector=self.config.max_per_sector
        )
        self.optimizer = WeightOptimizer()
        self.index_fetcher = IndexFetcher()
        self.date_calc = DateCalculator()
        self.returns_calc = ReturnsCalculator()
        self.metrics_calc = PortfolioMetricsCalculator()
    
    def analyze(
        self,
        candidate_tickers: List[str],
        start_date: str = '',
        end_date: str = ''
    ) -> Dict:

        start_date, end_date = self.date_calc.get_date_range(
            start_date, end_date, self.config.lookback_years
        )

        analysis_results = self._analyze_companies(candidate_tickers)
        
        if not analysis_results:
            return {'success': False, 'error': 'No se pudo analizar ninguna empresa'}

        selected_tickers = self.selector.select(
            analysis_results,
            method=self.config.selection_method
        )
        
        if not selected_tickers:
            return {'success': False, 'error': 'No hay empresas que cumplan criterios'}

        returns_data = None
        if self.config.weight_method == 'markowitz':
            hist_data = self.data_manager.download_assets(selected_tickers, start_date, end_date)
            returns_data = self.returns_calc.calculate(hist_data)

        weights = self.optimizer.optimize(
            selected_tickers,
            method=self.config.weight_method,
            returns_data=returns_data,
            analysis_results=analysis_results
        )

        metrics = self.metrics_calc.calculate(
            selected_tickers,
            weights,
            analysis_results
        )
        
        return {
            'success': True,
            'tickers': selected_tickers,
            'weights': weights,
            'metrics': metrics,
            'analysis': {t: analysis_results[t] for t in selected_tickers},
            'period': {'start': start_date, 'end': end_date}
        }
    
    def analyze_from_index(
        self,
        index_name: str,
        start_date: str = '',
        end_date: str = ''
    ) -> Dict:

        try:
            tickers = self.index_fetcher.get_index_components(index_name)
            if not tickers:
                return {
                    'success': False,
                    'error': f'No se pudieron obtener componentes de {index_name}'
                }
        except ValueError as e:
            return {'success': False, 'error': str(e)}

        return self.analyze(tickers, start_date, end_date)
    
    def analyze_from_etf(
        self,
        etf_ticker: str,
        start_date: str = '',
        end_date: str = ''
    ) -> Dict:

        tickers = self.index_fetcher.get_etf_holdings(etf_ticker)
        
        if not tickers:
            return {
                'success': False,
                'error': f'No se pudieron obtener holdings de {etf_ticker}'
            }

        return self.analyze(tickers, start_date, end_date)
    
    def _analyze_companies(self, tickers: List[str]) -> Dict:
        results = {}
        for ticker in tickers:
            try:
                result = self.company_analyzer.analyze(ticker)
                if result.get('success'):
                    results[ticker] = result
            except Exception as e:
                print(f"⚠️  {ticker}: {e}")
        return results