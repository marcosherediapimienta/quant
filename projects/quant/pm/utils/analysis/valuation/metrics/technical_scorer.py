import pandas as pd
import numpy as np
from .technical_calculator import TechnicalCalculator

class TechnicalScorer:

    def __init__(self):
        self.calculator = TechnicalCalculator()
    
    def score(self, hist: pd.DataFrame) -> float:

        if hist.empty or len(hist) < 21:
            return 50
        
        prices = hist['Close']
        rsi = self.calculator.calculate_rsi(prices)
        momentum = self.calculator.calculate_momentum(prices)
        
        score = 50
        
        if pd.notna(rsi):
            if rsi < 30:
                score += 20
            elif rsi > 70:
                score -= 20
        
        if pd.notna(momentum['1m']) and pd.notna(momentum['3m']):
            if momentum['1m'] > 5 and momentum['3m'] > 10:
                score += 20
            elif momentum['1m'] < -5 and momentum['3m'] < -10:
                score -= 20
        
        return np.clip(score, 0, 100)