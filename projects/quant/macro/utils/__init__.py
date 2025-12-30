"""
Utilidades para análisis macro.

Este módulo agrega componentes, analyzers, reporters y visualizations.
"""

# Components (calculadores de bajo nivel)
from .components import (
    MacroDataLoader,
    MacroDataDownloader,
    MacroTransformCalculator,
    MacroCorrelationCalculator,
    MacroRegressionCalculator,
    FactorCollinearityAnalyzer,
    RegressionResult,
    YieldCurveAnalysis,
    InflationSignals,
    CreditConditions,
    RiskSentiment,
)

# Analyzers (orquestadores de alto nivel)
from .analyzers import (
    MacroFactorAnalyzer,
    MacroCorrelationAnalyzer,
    MacroSensitivityAnalyzer,
    MacroSituationAnalyzer,
    FactorSelectionAnalyzer,
)

# Reporters (presentadores de resultados)
from .reporters import (
    MacroFactorReporter,
    MacroCorrelationReporter,
    MacroSensitivityReporter,
    MacroSituationReporter,
)

# Visualizations (plotters y visualizers)
from .visualizations import (
    FactorContributionsPlotter,
    FactorLoadingsPlotter,
    CorrelationPlotter,
    RollingBetasPlotter,
    YieldCurvePlotter,
    MacroFactorVisualizer,
    MacroCorrelationVisualizer,
    MacroSensitivityVisualizer,
    MacroSituationVisualizer,
)

# Tools (configuración)
from .tools import (
    MACRO_FACTORS,
    MACRO_CORE_FACTORS,
    MACRO_GLOBAL_FACTORS,
    FACTORS_TO_USE,
    ANNUAL_FACTOR,
    MAX_LAG,
)

__all__ = [
    # Components
    'MacroDataLoader',
    'MacroDataDownloader',
    'MacroTransformCalculator',
    'MacroCorrelationCalculator',
    'MacroRegressionCalculator',
    'FactorCollinearityAnalyzer',
    'RegressionResult',
    'YieldCurveAnalysis',
    'InflationSignals',
    'CreditConditions',
    'RiskSentiment',
    # Analyzers
    'MacroFactorAnalyzer',
    'MacroCorrelationAnalyzer',
    'MacroSensitivityAnalyzer',
    'MacroSituationAnalyzer',
    'FactorSelectionAnalyzer',
    # Reporters
    'MacroFactorReporter',
    'MacroCorrelationReporter',
    'MacroSensitivityReporter',
    'MacroSituationReporter',
    # Visualizations
    'FactorContributionsPlotter',
    'FactorLoadingsPlotter',
    'CorrelationPlotter',
    'RollingBetasPlotter',
    'YieldCurvePlotter',
    'MacroFactorVisualizer',
    'MacroCorrelationVisualizer',
    'MacroSensitivityVisualizer',
    'MacroSituationVisualizer',
    # Config
    'MACRO_FACTORS',
    'MACRO_CORE_FACTORS',
    'MACRO_GLOBAL_FACTORS',
    'FACTORS_TO_USE',
    'ANNUAL_FACTOR',
    'MAX_LAG',
]

