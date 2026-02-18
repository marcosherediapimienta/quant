import logging
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
from threading import Lock

from ...valuation.analyzers.company_analyzer import CompanyAnalyzer
from ....data import DataManager
from ..components.selector import CompanySelector
from ..components.optimizer import WeightOptimizer
from ..components.index_fetcher import IndexFetcher
from ..components.date_utils import DateCalculator
from ..components.returns_calculator import ReturnsCalculator
from ..components.metrics_calculator import PortfolioMetricsCalculator
from ....tools.config import PORTFOLIO_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    min_score: float = PORTFOLIO_CONFIG['selection']['min_score']
    max_companies: int = PORTFOLIO_CONFIG['selection']['max_companies']
    max_per_sector: int = PORTFOLIO_CONFIG['selection']['max_per_sector']
    selection_method: str = PORTFOLIO_CONFIG['selection']['default_method']
    weight_method: str = PORTFOLIO_CONFIG['optimization']['default_method']
    lookback_years: int = PORTFOLIO_CONFIG['dates']['default_lookback_years']
    start_date: str = PORTFOLIO_CONFIG['dates']['start_date']
    end_date: str = PORTFOLIO_CONFIG['dates']['end_date']

class PortfolioAnalyzer:
    def __init__(
        self, 
        config: PortfolioConfig = None,
        data_manager: DataManager = None
    ):

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
        self._rate_limit_lock = Lock()
        self._last_request_time = 0
        self._min_request_interval = 0.5
    
    def analyze(
        self,
        candidate_tickers: List[str],
        start_date: str = '',
        end_date: str = ''
    ) -> Dict:

        start_date, end_date = self._resolve_dates(start_date, end_date)

        if len(candidate_tickers) > 50:
            logger.info(
                "Phase 1: quick analysis of %d companies to identify best candidates "
                "(this may take several minutes due to Yahoo Finance rate limiting)...",
                len(candidate_tickers)
            )
            quick_results = self._analyze_companies_quick(candidate_tickers)
            valid_tickers = [
                ticker for ticker, result in quick_results.items()
                if result.get('success') and result.get('scores', {}).get('total', 0) >= self.config.min_score
            ]

            valid_tickers.sort(
                key=lambda t: quick_results[t].get('scores', {}).get('total', 0),
                reverse=True
            )
            
            logger.info("Phase 1 complete — %d companies pass min_score=%.1f", len(valid_tickers), self.config.min_score)

            if len(valid_tickers) == 0:
                success_count = sum(1 for r in quick_results.values() if r.get('success'))
                fail_count = len(quick_results) - success_count
                logger.warning(
                    "No companies passed the score threshold — successful: %d/%d, failed: %d/%d",
                    success_count, len(quick_results), fail_count, len(quick_results)
                )
                
                errors = [r.get('error', 'Unknown') for r in quick_results.values() if not r.get('success')]
                from collections import Counter
                error_counts = Counter(errors)
                for error, count in error_counts.most_common(5):
                    logger.warning("  Common error (%dx): %s", count, error)
                
                all_scores = [
                    (ticker, result.get('scores', {}).get('total', 0)) 
                    for ticker, result in quick_results.items() 
                    if result.get('success')
                ]
                all_scores.sort(key=lambda x: x[1], reverse=True)
                logger.warning("Top available scores: %s", all_scores[:10])
                logger.warning("Required min_score: %s", self.config.min_score)

            max_candidates = max(self.config.max_companies * 3, 50)
            top_candidates = valid_tickers[:max_candidates]
            
            logger.info("Phase 2: full analysis of %d top candidates...", len(top_candidates))
            analysis_results = self._analyze_companies(top_candidates)
        else:
            analysis_results = self._analyze_companies(candidate_tickers)
        
        if not analysis_results:
            logger.error("Could not analyze any company")
            return {'success': False, 'error': 'No se pudo analizar ninguna empresa'}

        logger.info("%d companies analyzed successfully", len(analysis_results))
        logger.info(
            "Selecting companies — method: %s, min_score: %.1f, max: %d",
            self.config.selection_method, self.config.min_score, self.config.max_companies
        )
        
        selected_tickers = self.selector.select(
            analysis_results,
            method=self.config.selection_method
        )
        
        if not selected_tickers:
            available = [(t, analysis_results[t].get('scores', {}).get('total', 0)) for t in list(analysis_results.keys())[:5]]
            logger.error("No companies meet criteria. Sample scores: %s", available)
            return {'success': False, 'error': 'No hay empresas que cumplan criterios'}

        logger.info("Selected %d companies: %s", len(selected_tickers), ', '.join(selected_tickers))

        returns_data = None
        methods_requiring_returns = ('markowitz', 'risk_parity', 'score_risk_adjusted', 'black_litterman')
        if self.config.weight_method in methods_requiring_returns:
            logger.info("Downloading historical data for %d companies...", len(selected_tickers))
            try:
                hist_data = self.data_manager.download_assets(
                    selected_tickers, 
                    start_date, 
                    end_date,
                    progress=False  
                )
                logger.debug("Calculating historical returns...")
                returns_data = self.returns_calc.calculate(hist_data)
                if returns_data is None or returns_data.empty:
                    logger.warning("Could not calculate returns, falling back to equal weights")
                    returns_data = None
            except Exception as e:
                logger.warning("Error downloading historical data: %s — using equal weights", e)
                returns_data = None

        logger.info("Optimizing portfolio weights (method: %s)...", self.config.weight_method)
        weights = self.optimizer.optimize(
            selected_tickers,
            method=self.config.weight_method,
            returns_data=returns_data,
            analysis_results=analysis_results
        )

        logger.info("Calculating portfolio metrics...")
        metrics = self.metrics_calc.calculate(
            selected_tickers,
            weights,
            analysis_results
        )
        
        logger.info("Portfolio built successfully with %d companies", len(selected_tickers))
        selected_analysis = {t: analysis_results[t] for t in selected_tickers}
        
        return {
            'success': True,
            'tickers': selected_tickers,
            'weights': weights,
            'metrics': metrics,
            'analysis': selected_analysis,
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
    
    def _resolve_dates(self, start_date: str, end_date: str) -> tuple[str, str]:

        if not start_date:
            start_date = self.config.start_date
        
        if not end_date:
            end_date = self.config.end_date

        start_date, end_date = self.date_calc.get_date_range(
            start_date, 
            end_date, 
            self.config.lookback_years
        )
        
        return start_date, end_date
    
    def _rate_limit(self):
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._min_request_interval:
                time.sleep(self._min_request_interval - time_since_last)
            self._last_request_time = time.time()
    
    def _analyze_companies_quick(self, tickers: List[str]) -> Dict:

        if not tickers:
            return {}
            
        results = {}
        max_workers = max(1, min(os.cpu_count() or 4, len(tickers), 10)) 
        
        def analyze_quick_single(ticker: str) -> tuple[str, Dict]:
            self._rate_limit()
            try:
                result = self.company_analyzer.analyze_quick(ticker)
                error_str = str(result.get('error', ''))

                if not result.get('success') and ('401' in error_str or 'Rate limit' in error_str or 'Too Many Requests' in error_str):
                    time.sleep(2)  
                    result = self.company_analyzer.analyze_quick(ticker)
                return (ticker, result)
            except Exception as e:
                error_msg = str(e)

                if '401' in error_msg or 'Unauthorized' in error_msg or 'Rate limit' in error_msg or 'Too Many Requests' in error_msg:
                    time.sleep(2)
                    try:
                        result = self.company_analyzer.analyze_quick(ticker)
                        return (ticker, result)
                    except Exception:
                        pass
                logger.warning("Quick analysis failed for %s: %s", ticker, error_msg)
                return (ticker, {'success': False, 'error': error_msg})

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(analyze_quick_single, ticker): ticker for ticker in tickers}
            
            completed = 0
            successful = 0
            total = len(tickers)

            for future in as_completed(future_to_ticker):
                ticker, result = future.result()
                completed += 1
                results[ticker] = result

                if result.get('success'):
                    successful += 1

                if completed % 50 == 0 or completed == total:
                    logger.info("Quick analysis: %d/%d companies (%d successful)...", completed, total, successful)
        
        return results
    
    def _analyze_companies(self, tickers: List[str]) -> Dict:

        if not tickers:
            return {}
            
        results = {}
        max_workers = max(1, min(3, len(tickers)))

        def analyze_single(ticker: str) -> tuple[str, Dict]:
            self._rate_limit()
            try:
                result = self.company_analyzer.analyze(ticker)

                if not result.get('success') and '401' in str(result.get('error', '')):
                    time.sleep(1)
                    result = self.company_analyzer.analyze(ticker)
                return (ticker, result)
            except Exception as e:
                error_msg = str(e)

                if '401' in error_msg or 'Unauthorized' in error_msg:
                    time.sleep(1)

                    try:
                        result = self.company_analyzer.analyze(ticker)
                        return (ticker, result)
                    except Exception:
                        pass

                logger.warning("Full analysis failed for %s: %s", ticker, error_msg)
                return (ticker, {'success': False, 'error': error_msg})

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(analyze_single, ticker): ticker for ticker in tickers}
            completed = 0
            total = len(tickers)

            for future in as_completed(future_to_ticker):
                ticker, result = future.result()
                completed += 1

                if result.get('success'):
                    results[ticker] = result

                if completed % 10 == 0 or completed == total:
                    logger.info("Analysis progress: %d/%d companies", completed, total)
                    
        return results
