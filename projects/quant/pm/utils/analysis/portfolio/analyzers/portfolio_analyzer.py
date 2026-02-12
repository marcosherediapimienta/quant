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
            print(f"⚡ Fase 1: Análisis rápido de todas las {len(candidate_tickers)} empresas para identificar mejores candidatas...")
            print(f"   ⏱️  Esto puede tomar varios minutos debido a rate limiting de Yahoo Finance...")
            quick_results = self._analyze_companies_quick(candidate_tickers)
            valid_tickers = [
                ticker for ticker, result in quick_results.items()
                if result.get('success') and result.get('scores', {}).get('total', 0) >= self.config.min_score
            ]

            valid_tickers.sort(
                key=lambda t: quick_results[t].get('scores', {}).get('total', 0),
                reverse=True
            )
            
            print(f"   ✅ {len(valid_tickers)} empresas superan el min_score de {self.config.min_score}")

            if len(valid_tickers) == 0:
                success_count = sum(1 for r in quick_results.values() if r.get('success'))
                fail_count = len(quick_results) - success_count
                print(f"   ⚠️  Empresas exitosas: {success_count}/{len(quick_results)}")
                print(f"   ⚠️  Empresas fallidas: {fail_count}/{len(quick_results)}")
                
                errors = [r.get('error', 'Unknown') for r in quick_results.values() if not r.get('success')]
                from collections import Counter
                error_counts = Counter(errors)
                print(f"   ⚠️  Errores más comunes:")
                for error, count in error_counts.most_common(5):
                    print(f"      {error}: {count}")
                
                all_scores = [
                    (ticker, result.get('scores', {}).get('total', 0)) 
                    for ticker, result in quick_results.items() 
                    if result.get('success')
                ]
                all_scores.sort(key=lambda x: x[1], reverse=True)
                print(f"   ⚠️  Top 10 scores disponibles:")
                for ticker, score in all_scores[:10]:
                    print(f"      {ticker}: {score:.2f}")
                print(f"   ⚠️  Min score requerido: {self.config.min_score}")

            max_candidates = max(self.config.max_companies * 3, 50)
            top_candidates = valid_tickers[:max_candidates]
            
            print(f"⚡ Fase 2: Análisis completo de {len(top_candidates)} mejores candidatas...")
            analysis_results = self._analyze_companies(top_candidates)
        else:
            analysis_results = self._analyze_companies(candidate_tickers)
        
        if not analysis_results:
            print(f"❌ Error: No se pudo analizar ninguna empresa")
            return {'success': False, 'error': 'No se pudo analizar ninguna empresa'}

        print(f"✅ Se analizaron {len(analysis_results)} empresas correctamente")
        print(f"🎯 Seleccionando mejores empresas según criterios...")
        print(f"   - Método: {self.config.selection_method}")
        print(f"   - Min score: {self.config.min_score}")
        print(f"   - Max companies: {self.config.max_companies}")
        
        selected_tickers = self.selector.select(
            analysis_results,
            method=self.config.selection_method
        )
        
        if not selected_tickers:
            print(f"❌ Error: No hay empresas que cumplan criterios")
            print(f"   Scores disponibles: {[(t, analysis_results[t].get('scores', {}).get('total', 0)) for t in list(analysis_results.keys())[:5]]}")
            return {'success': False, 'error': 'No hay empresas que cumplan criterios'}
        print(f"✅ Seleccionadas {len(selected_tickers)} empresas: {', '.join(selected_tickers)}")

        returns_data = None
        methods_requiring_returns = ('markowitz', 'score_risk_adjusted', 'black_litterman')
        if self.config.weight_method in methods_requiring_returns:
            print(f"📥 Descargando datos históricos para {len(selected_tickers)} empresas...")
            try:
                hist_data = self.data_manager.download_assets(
                    selected_tickers, 
                    start_date, 
                    end_date,
                    progress=False  
                )
                print(f"📊 Calculando retornos históricos...")
                returns_data = self.returns_calc.calculate(hist_data)
                if returns_data is None or returns_data.empty:
                    print(f"⚠️  No se pudieron calcular retornos, usando pesos iguales")
                    returns_data = None
            except Exception as e:
                print(f"⚠️  Error descargando datos históricos: {e}")
                print(f"⚠️  Usando pesos iguales como fallback")
                returns_data = None

        print(f"⚖️  Optimizando pesos del portfolio (método: {self.config.weight_method})...")
        weights = self.optimizer.optimize(
            selected_tickers,
            method=self.config.weight_method,
            returns_data=returns_data,
            analysis_results=analysis_results
        )

        print(f"📈 Calculando métricas del portfolio...")
        metrics = self.metrics_calc.calculate(
            selected_tickers,
            weights,
            analysis_results
        )
        
        print(f"✅ Portfolio construido exitosamente con {len(selected_tickers)} empresas")
        print(f"📦 Preparando respuesta...")
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
                    except:
                        pass
                print(f"⚠️  {ticker}: {error_msg}")
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
                    print(f"📊 Análisis rápido: {completed}/{total} empresas ({successful} exitosas)...")
        
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
                    except:
                        pass

                print(f"⚠️  {ticker}: {error_msg}")
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
                    print(f"📊 Progreso: {completed}/{total} empresas analizadas...")
                    
        return results