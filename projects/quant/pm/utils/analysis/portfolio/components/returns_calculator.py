import pandas as pd
import numpy as np
from ....tools.config import CLOSE_COL

class ReturnsCalculator:
    def __init__(self, price_column: str = CLOSE_COL):
        self.price_column = price_column
    
    def extract_prices(self, data: pd.DataFrame) -> pd.DataFrame:

        if isinstance(data.columns, pd.MultiIndex):
            if self.price_column in data.columns.get_level_values(0):
                return data[self.price_column]
        
        if self.price_column in data.columns:
            return data[self.price_column]
        return data
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self.extract_prices(data)
        return prices.pct_change().dropna()
    
    def calculate_log_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        prices = self.extract_prices(data)
        return np.log(prices / prices.shift(1)).dropna()