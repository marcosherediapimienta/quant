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
    def __init__(
        self, 
        data_manager: DataManager = None,
        start_date: str = None,
        end_date: str = None,
        lookback_years: int = None
    ):

        self.company_analyzer = CompanyAnalyzer()
        self.data_manager = data_manager if data_manager else DataManager()
        self.date_calc = DateCalculator()

        config = TRADING_SIGNALS_CONFIG
        self.start_date = start_date if start_date is not None else config['start_date']
        self.end_date = end_date if end_date is not None else config['end_date']
        self.use_current_date = config['use_current_date_as_end']
        self.lookback_years = lookback_years if lookback_years else config['default_lookback_years']
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

        company_data = self.company_analyzer.fetch_data(ticker)
        if not company_data.get('success'):
            raise ValueError(f"Error obteniendo datos: {company_data.get('error')}")

        analysis = self.company_analyzer.analyze(ticker, company_data['data'])
        final_start, final_end = self._resolve_dates(start_date, end_date)
        current_price = self._get_current_price(ticker, final_start, final_end, company_data)
        scores = self._extract_scores(analysis)
        signal, confidence = self.signal_determiner.determine(
            scores['valuation'], 
            scores['fundamental']
        )

        price_target = self.price_target.calculate(
            company_data['data'], 
            scores['valuation'], 
            current_price
        )
        upside = self._calculate_upside(price_target, current_price)

        # Sanity check: evitar contradicciones entre señal y upside
        signal, confidence = self.signal_determiner.validate_with_upside(
            signal, confidence, upside
        )

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
        final_start = start_date if start_date else self.start_date
        final_end = end_date if end_date else self.end_date

        if not final_start:
            final_start = self.date_calc.get_lookback_date_from_years(self.lookback_years)

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

        try:
            hist = self.data_manager.download_assets([ticker], start_date, end_date)
            
            if not hist.empty and ticker in hist.columns:
                return float(hist[ticker].iloc[-1])
        except Exception:
            pass 
 
        current_price = company_data['data'].get('currentPrice', 0)
        if current_price == 0:
            current_price = company_data['data'].get('regularMarketPrice', 0)
        
        return float(current_price)
    
    def _extract_scores(self, analysis: dict) -> dict:
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

        if current_price <= 0:
            return 0.0
            
        return (price_target / current_price) - 1