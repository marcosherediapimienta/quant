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
from ....tools.config import PORTFOLIO_CONFIG

@dataclass
class PortfolioConfig:
    """Configuración del analizador de portfolio."""
    min_score: float = PORTFOLIO_CONFIG['selection']['min_score']
    max_companies: int = PORTFOLIO_CONFIG['selection']['max_companies']
    max_per_sector: int = PORTFOLIO_CONFIG['selection']['max_per_sector']
    selection_method: str = PORTFOLIO_CONFIG['selection']['default_method']
    weight_method: str = PORTFOLIO_CONFIG['optimization']['default_method']
    lookback_years: int = PORTFOLIO_CONFIG['dates']['default_lookback_years']
    start_date: str = PORTFOLIO_CONFIG['dates']['start_date']
    end_date: str = PORTFOLIO_CONFIG['dates']['end_date']

class PortfolioAnalyzer:
    """
    Analizador de portfolios que coordina la selección de empresas,
    optimización de pesos y cálculo de métricas.
    
    Responsabilidad única: Coordinar el análisis completo de portfolio.
    """

    def __init__(
        self, 
        config: PortfolioConfig = None,
        data_manager: DataManager = None
    ):
        """
        Inicializa el analizador con configuración y dependencias.
        
        Args:
            config: Configuración del portfolio (usa defaults de config.py si None)
            data_manager: Gestor de datos compartido (crea uno nuevo si None)
        """
        self.config = config if config else PortfolioConfig()
        self.data_manager = data_manager if data_manager else DataManager()
        
        # Componentes especializados (cada uno con su responsabilidad única)
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
        """
        Analiza un conjunto de tickers y construye un portfolio optimizado.
        
        Args:
            candidate_tickers: Lista de tickers candidatos
            start_date: Fecha inicial (vacío = usar config)
            end_date: Fecha final (vacío = usar config)
            
        Returns:
            Dict con resultados del análisis o error
        """
        # Calcular fechas desde config si no se especifican
        start_date, end_date = self._resolve_dates(start_date, end_date)

        # Paso 1: Analizar empresas candidatas
        analysis_results = self._analyze_companies(candidate_tickers)
        if not analysis_results:
            return {'success': False, 'error': 'No se pudo analizar ninguna empresa'}

        # Paso 2: Seleccionar mejores empresas según criterios
        selected_tickers = self.selector.select(
            analysis_results,
            method=self.config.selection_method
        )
        if not selected_tickers:
            return {'success': False, 'error': 'No hay empresas que cumplan criterios'}

        # Paso 3: Descargar datos históricos si se necesita optimización Markowitz
        returns_data = None
        if self.config.weight_method == 'markowitz':
            hist_data = self.data_manager.download_assets(
                selected_tickers, 
                start_date, 
                end_date
            )
            returns_data = self.returns_calc.calculate(hist_data)

        # Paso 4: Optimizar pesos del portfolio
        weights = self.optimizer.optimize(
            selected_tickers,
            method=self.config.weight_method,
            returns_data=returns_data,
            analysis_results=analysis_results
        )

        # Paso 5: Calcular métricas del portfolio
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
        """
        Analiza un portfolio basado en componentes de un índice.
        
        Args:
            index_name: Nombre del índice (ej: 'SP500', 'NASDAQ100')
            start_date: Fecha inicial (vacío = usar config)
            end_date: Fecha final (vacío = usar config)
            
        Returns:
            Dict con resultados del análisis o error
        """
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
        """
        Analiza un portfolio basado en holdings de un ETF.
        
        Args:
            etf_ticker: Ticker del ETF (ej: 'SPY', 'QQQ')
            start_date: Fecha inicial (vacío = usar config)
            end_date: Fecha final (vacío = usar config)
            
        Returns:
            Dict con resultados del análisis o error
        """
        tickers = self.index_fetcher.get_etf_holdings(etf_ticker)
        
        if not tickers:
            return {
                'success': False,
                'error': f'No se pudieron obtener holdings de {etf_ticker}'
            }

        return self.analyze(tickers, start_date, end_date)
    
    def _resolve_dates(self, start_date: str, end_date: str) -> tuple[str, str]:
        """
        Resuelve fechas usando config si no se especifican.
        
        Responsabilidad: Centralizar la lógica de resolución de fechas.
        """
        if not start_date:
            start_date = self.config.start_date
        
        if not end_date:
            end_date = self.config.end_date
        
        # Si aún están vacías, calcular desde lookback_years
        start_date, end_date = self.date_calc.get_date_range(
            start_date, 
            end_date, 
            self.config.lookback_years
        )
        
        return start_date, end_date
    
    def _analyze_companies(self, tickers: List[str]) -> Dict:
        """
        Analiza fundamentalmente cada empresa.
        
        Responsabilidad: Delegar análisis de empresa al CompanyAnalyzer.
        """
        results = {}
        for ticker in tickers:
            try:
                result = self.company_analyzer.analyze(ticker)
                if result.get('success'):
                    results[ticker] = result
            except Exception as e:
                print(f"⚠️  {ticker}: {e}")
        return results