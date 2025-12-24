import numpy as np
from typing import Dict
from dataclasses import dataclass

from ..metrics import (
    ProfitabilityMetrics,
    FinancialHealthMetrics,
    GrowthMetrics,
    EfficiencyMetrics,
    ValuationMultiples,
    ProfitabilityThresholds,
    FinancialHealthThresholds,
    GrowthThresholds,
    EfficiencyThresholds,
    ValuationThresholds
)
from ....tools.config import COMPANY_ANALYSIS_WEIGHTS, CONCLUSION_THRESHOLDS

@dataclass
class AnalysisWeights:
    """Pesos para combinar scores de diferentes categorías."""
    profitability: float = None
    financial_health: float = None
    growth: float = None
    efficiency: float = None
    valuation: float = None
    
    def __post_init__(self):
        # Cargar defaults desde config si no se proporcionan
        defaults = COMPANY_ANALYSIS_WEIGHTS['default']
        self.profitability = self.profitability if self.profitability is not None else defaults['profitability']
        self.financial_health = self.financial_health if self.financial_health is not None else defaults['financial_health']
        self.growth = self.growth if self.growth is not None else defaults['growth']
        self.efficiency = self.efficiency if self.efficiency is not None else defaults['efficiency']
        self.valuation = self.valuation if self.valuation is not None else defaults['valuation']
        
        # Normalizar si no suman 1.0
        total = (self.profitability + self.financial_health + 
                 self.growth + self.efficiency + self.valuation)
        if not np.isclose(total, 1.0):
            self.profitability /= total
            self.financial_health /= total
            self.growth /= total
            self.efficiency /= total
            self.valuation /= total

@dataclass
class ConclusionThresholds:
    """Umbrales para clasificación de scores."""
    excellent: float = None
    good: float = None
    fair: float = None
    weak: float = None
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        # Cargar desde config si no se proporciona
        cfg = CONCLUSION_THRESHOLDS
        self.excellent = self.excellent if self.excellent is not None else cfg['excellent']
        self.good = self.good if self.good is not None else cfg['good']
        self.fair = self.fair if self.fair is not None else cfg['fair']
        self.weak = self.weak if self.weak is not None else cfg['weak']
        self.labels = self.labels or cfg['labels'].copy()

class CompanyAnalyzer:
    """
    Analizador fundamental de empresas.
    
    Responsabilidad: Coordinar análisis completo de fundamentales de una empresa.
    """

    def __init__(
        self,
        weights: AnalysisWeights = None,
        conclusion_thresholds: ConclusionThresholds = None,
        profitability_thresholds: ProfitabilityThresholds = None,
        health_thresholds: FinancialHealthThresholds = None,
        growth_thresholds: GrowthThresholds = None,
        efficiency_thresholds: EfficiencyThresholds = None,
        valuation_thresholds: ValuationThresholds = None
    ):
        self.weights = weights or AnalysisWeights()
        self.conclusion_thresholds = conclusion_thresholds or ConclusionThresholds()
        self.profitability = ProfitabilityMetrics(profitability_thresholds)
        self.health = FinancialHealthMetrics(health_thresholds)
        self.growth = GrowthMetrics(growth_thresholds)
        self.efficiency = EfficiencyMetrics(efficiency_thresholds)
        self.valuation = ValuationMultiples(valuation_thresholds)

# ... (resto del código sin cambios) ...