from dataclasses import dataclass
from typing import Optional

from ..analyzers.company_analyzer import CompanyAnalyzer
from ....data import DataManager
from ....tools.config import TRADING_SIGNALS_CONFIG
from ...portfolio.components.date_utils import DateCalculator

from ..metrics.score_extractor import ScoreExtractor
from ..metrics.fundamental_aggregator import FundamentalAggregator
from ..metrics.signal_determiner import SignalDeterminer
from ..metrics.price_target_calculator import PriceTargetCalculator
from ..metrics.reason_generator import ReasonGenerator

@dataclass
class TradingSignal:
    """Señal de trading con toda la información relevante."""
    ticker: str
    signal: str
    confidence: float
    valuation_score: float
    fundamental_score: float
    current_price: float
    price_target: float
    upside_potential: float
    reasons: list
    technical_score: Optional[float] = None 
    
class BuySellSignalsAnalyzer:
    """
    Analizador de señales de compra/venta basado en análisis fundamental.
    
    Responsabilidad única: Generar señales de trading coordinando
    análisis fundamental, valoración y precio objetivo.
    """

    def __init__(
        self, 
        data_manager: DataManager = None,
        start_date: str = None,
        end_date: str = None,
        lookback_years: int = None
    ):
        """
        Inicializa el analizador de señales.
        
        Args:
            data_manager: Gestor de datos compartido (crea uno si None)
            start_date: Fecha inicial (None = usar config.py)
            end_date: Fecha final (None = usar config.py)
            lookback_years: Años hacia atrás (None = usar config.py)
        """
        # Componentes con responsabilidades específicas
        self.company_analyzer = CompanyAnalyzer()
        self.data_manager = data_manager if data_manager else DataManager()
        self.date_calc = DateCalculator()
        
        # Configuración desde config.py con posibilidad de override
        config = TRADING_SIGNALS_CONFIG
        self.start_date = start_date if start_date is not None else config['start_date']
        self.end_date = end_date if end_date is not None else config['end_date']
        self.use_current_date = config['use_current_date_as_end']
        self.lookback_years = lookback_years if lookback_years else config['default_lookback_years']
        
        # Componentes de análisis especializados
        self.score_extractor = ScoreExtractor()
        self.fundamental_agg = FundamentalAggregator()
        self.signal_determiner = SignalDeterminer()
        self.price_target = PriceTargetCalculator()
        self.reason_gen = ReasonGenerator()
    
    def analyze_stock(
        self, 
        ticker: str,
        start_date: str = None,
        end_date: str = None
    ) -> TradingSignal:
        """
        Analiza una acción y genera señales de trading.
        
        Args:
            ticker: Símbolo del ticker
            start_date: Override fecha inicial (None = usar configuración)
            end_date: Override fecha final (None = usar configuración)
            
        Returns:
            TradingSignal con toda la información de la señal
            
        Raises:
            ValueError: Si hay error en el análisis de la empresa
        """
        # Paso 1: Obtener datos fundamentales de la empresa
        company_data = self.company_analyzer.fetch_data(ticker)
        if not company_data.get('success'):
            raise ValueError(f"Error obteniendo datos: {company_data.get('error')}")

        # Paso 2: Analizar fundamentalmente la empresa
        analysis = self.company_analyzer.analyze(ticker, company_data['data'])
        
        # Paso 3: Obtener precio actual desde datos históricos
        final_start, final_end = self._resolve_dates(start_date, end_date)
        current_price = self._get_current_price(ticker, final_start, final_end, company_data)
        
        # Paso 4: Extraer scores de análisis
        scores = self._extract_scores(analysis)
        
        # Paso 5: Determinar señal y confianza
        signal, confidence = self.signal_determiner.determine(
            scores['valuation'], 
            scores['fundamental']
        )

        # Paso 6: Calcular precio objetivo y potencial
        price_target = self.price_target.calculate(
            company_data['data'], 
            scores['valuation'], 
            current_price
        )
        upside = self._calculate_upside(price_target, current_price)
        
        # Paso 7: Generar razones de la señal
        reasons = self.reason_gen.generate(analysis, scores['fundamental'], None)
        
        return TradingSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            valuation_score=scores['valuation'],
            fundamental_score=scores['fundamental'],
            current_price=current_price,
            price_target=price_target,
            upside_potential=upside,
            reasons=reasons,
            technical_score=None
        )
    
    def _resolve_dates(self, start_date: str, end_date: str) -> tuple[str, str]:
        """
        Resuelve fechas usando configuración si no se especifican.
        
        Responsabilidad: Centralizar lógica de fechas.
        """
        # Usar parámetros o configuración de instancia
        final_start = start_date if start_date else self.start_date
        final_end = end_date if end_date else self.end_date
        
        # Si start_date está vacío, calcular desde lookback_years
        if not final_start:
            final_start = self.date_calc.get_lookback_date_from_years(self.lookback_years)
        
        # Si use_current_date está activo, usar fecha actual como final
        if self.use_current_date or not final_end:
            final_end = self.date_calc.get_current_date_str()
        
        return final_start, final_end
    
    def _get_current_price(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        company_data: dict
    ) -> float:
        """
        Obtiene el precio actual del ticker.
        
        Responsabilidad: Abstraer la obtención de precio con fallbacks.
        
        Prioridad:
        1. Último precio de datos históricos
        2. currentPrice de yfinance
        3. regularMarketPrice de yfinance
        """
        try:
            hist = self.data_manager.download_assets([ticker], start_date, end_date)
            
            if not hist.empty and ticker in hist.columns:
                return float(hist[ticker].iloc[-1])
        except Exception:
            pass  # Fallar silenciosamente y usar fallback
        
        # Fallback a datos de yfinance
        current_price = company_data['data'].get('currentPrice', 0)
        if current_price == 0:
            current_price = company_data['data'].get('regularMarketPrice', 0)
        
        return float(current_price)
    
    def _extract_scores(self, analysis: dict) -> dict:
        """
        Extrae todos los scores del análisis.
        
        Responsabilidad: Centralizar extracción de scores.
        """
        return {
            'valuation': self.score_extractor.extract_valuation(analysis),
            'profitability': self.score_extractor.extract_profitability(analysis),
            'health': self.score_extractor.extract_health(analysis),
            'growth': self.score_extractor.extract_growth(analysis),
            'fundamental': self.fundamental_agg.aggregate(
                self.score_extractor.extract_profitability(analysis),
                self.score_extractor.extract_health(analysis),
                self.score_extractor.extract_growth(analysis)
            )
        }
    
    def _calculate_upside(self, price_target: float, current_price: float) -> float:
        """
        Calcula el potencial de subida en porcentaje (como decimal).
        
        Responsabilidad: Cálculo simple aislado.
        
        Returns:
            Potencial como decimal (0.1123 = 11.23%), no como porcentaje
        """
        if current_price <= 0:
            return 0.0
        # Retornar como decimal (0.1123 = 11.23%), el frontend lo convertirá a porcentaje
        return (price_target / current_price) - 1