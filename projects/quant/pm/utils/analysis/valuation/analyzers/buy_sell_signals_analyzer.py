from dataclasses import dataclass
from typing import Optional

from ..analyzers.company_analyzer import CompanyAnalyzer
from ....data import DataManager
from ....tools.config import TRADING_SIGNALS_CONFIG

from ..metrics.score_extractor import ScoreExtractor
from ..metrics.fundamental_aggregator import FundamentalAggregator
from ..metrics.signal_determiner import SignalDeterminer
from ..metrics.price_target_calculator import PriceTargetCalculator
from ..metrics.reason_generator import ReasonGenerator

from ...portfolio.components.date_utils import DateCalculator

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
        lookback_years: int = None
    ):
        self.company_analyzer = CompanyAnalyzer()
        self.data_manager = data_manager if data_manager else DataManager()
        self.date_calc = DateCalculator()
        self.lookback_years = (
            lookback_years if lookback_years 
            else TRADING_SIGNALS_CONFIG['default_lookback_years']
        )
        
        self.score_extractor = ScoreExtractor()
        self.fundamental_agg = FundamentalAggregator()
        self.signal_determiner = SignalDeterminer()
        self.price_target = PriceTargetCalculator()
        self.reason_gen = ReasonGenerator()
    
    def analyze_stock(
        self, 
        ticker: str,
        start_date: str = '',
        end_date: str = ''
    ) -> TradingSignal:

        company_data = self.company_analyzer.fetch_data(ticker)
        if not company_data.get('success'):
            raise ValueError(f"Error: {company_data.get('error')}")

        analysis = self.company_analyzer.analyze(ticker, company_data['data'])

        start_date, end_date = self.date_calc.get_date_range(
            start_date, 
            end_date, 
            self.lookback_years
        )
 
        hist = self.data_manager.download_assets([ticker], start_date, end_date)
        
        if not hist.empty:
            if ticker in hist.columns:
                current_price = float(hist[ticker].iloc[-1])
            else:
                current_price = company_data['data'].get('currentPrice', 0)
        else:
            current_price = company_data['data'].get('currentPrice', 0)
            if current_price == 0:
                current_price = company_data['data'].get('regularMarketPrice', 0)
        
        val_score = self.score_extractor.extract_valuation(analysis)
        prof_score = self.score_extractor.extract_profitability(analysis)
        health_score = self.score_extractor.extract_health(analysis)
        growth_score = self.score_extractor.extract_growth(analysis)
        fund_score = self.fundamental_agg.aggregate(prof_score, health_score, growth_score)
        signal, confidence = self.signal_determiner.determine(
            val_score, 
            fund_score
        )

        price_target = self.price_target.calculate(
            company_data['data'], 
            val_score, 
            current_price
        )

        upside = ((price_target / current_price) - 1) * 100 if current_price > 0 else 0
        reasons = self.reason_gen.generate(analysis, fund_score, None)
        
        return TradingSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            valuation_score=val_score,
            fundamental_score=fund_score,
            current_price=current_price,
            price_target=price_target,
            upside_potential=upside,
            reasons=reasons,
            technical_score=None
        )