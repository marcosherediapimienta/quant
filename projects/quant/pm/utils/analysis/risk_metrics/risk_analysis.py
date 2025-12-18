from .components.var import VaRCalculator
from .components.es import ESCalculator
from .components.sharpe import SharpeCalculator
from .components.sortino import SortinoCalculator
from .components.momentum import DistributionMoments


class RiskAnalysis:
    
    def __init__(self, annual_factor: float = 252.0):
        self.annual_factor = annual_factor
        self.var = VaRCalculator(annual_factor)
        self.es = ESCalculator(annual_factor)
        self.sharpe = SharpeCalculator(annual_factor)
        self.sortino = SortinoCalculator(annual_factor)
        self.moments = DistributionMoments()