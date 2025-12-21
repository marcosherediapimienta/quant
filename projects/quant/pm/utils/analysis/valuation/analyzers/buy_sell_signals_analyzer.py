from dataclasses import dataclass
import pandas as pd
from typing import Optional

from ..analyzers.company_analyzer import CompanyAnalyzer
from ....data.components.data_loader import DataLoader

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

    def __init__(self):
        self.company_analyzer = CompanyAnalyzer()
        self.data_loader = DataLoader()
        self.score_extractor = ScoreExtractor()
        self.fundamental_agg = FundamentalAggregator()
        self.signal_determiner = SignalDeterminer()
        self.price_target = PriceTargetCalculator()
        self.reason_gen = ReasonGenerator()
    
    def analyze_stock(self, ticker: str) -> TradingSignal:
        company_data = self.company_analyzer.fetch_data(ticker)
        if not company_data.get('success'):
            raise ValueError(f"Error: {company_data.get('error')}")

        analysis = self.company_analyzer.analyze(ticker, company_data['data'])

        hist = self.data_loader.download_single(ticker, "2020-01-01", "2024-12-31")
        
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                if 'Close' in hist.columns.get_level_values(0):
                    close_prices = hist['Close'].iloc[:, 0] if hist['Close'].ndim > 1 else hist['Close']
                    current_price = float(close_prices.iloc[-1])
                else:
                    current_price = company_data['data'].get('currentPrice', 0)
            elif 'Close' in hist.columns:
                current_price = float(hist['Close'].iloc[-1])
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