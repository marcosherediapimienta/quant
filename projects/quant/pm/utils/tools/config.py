# ============================================================================
# BENCHMARKS
# ============================================================================
BENCHMARKS = {
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'DOW': '^DJI',
    'MSCI_WORLD': 'EUNL.DE',
    'STOXX50': '^STOXX50E',
    'VIX': '^VIX',
    'RUSSELL2000': '^RUT',
    'FTSE100': '^FTSE',
    'DAX': '^GDAXI',
    'NIKKEI': '^N225',
    'CAC40': '^FCHI',
    'EUROSTOXX': '^STOXX',
    'IBEX35': '^IBEX'
}

BENCHMARK_CURRENCIES = {
    'SP500': 'USD',
    'NASDAQ': 'USD',
    'DOW': 'USD',
    'MSCI_WORLD': 'EUR',
    'STOXX50': 'EUR',
    'VIX': 'USD',
    'RUSSELL2000': 'USD',
    'FTSE100': 'GBP',
    'DAX': 'EUR',
    'NIKKEI': 'JPY',
    'CAC40': 'EUR',
    'EUROSTOXX': 'EUR',
    'IBEX35': 'EUR'
}

# ============================================================================
# CONFIGURACIÓN DE DESCARGA
# ============================================================================
DOWNLOAD_DEFAULTS = {
    'auto_adjust': True,
    'group_by': 'ticker',
    'threads': True,
    'progress': True
}

YFINANCE_COLUMNS = {
    'adj_close': 'Adj Close',
    'close': 'Close',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'volume': 'Volume',
}

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS DE RIESGO
# ============================================================================
RISK_ANALYSIS = {
    'annual_factor': 252,
    'risk_free_rate': 0.045,
    'default_confidence_levels': (0.90, 0.95, 0.99),
    'default_confidence_level': 0.95,
    'monte_carlo': {
        'n_simulations': 10000,
        'seed': 42
    },
    'rolling': {
        'default_window': 252
    }
}

# ============================================================================
# CONFIGURACIÓN ESTADÍSTICA
# ============================================================================
STATISTICAL_DEFAULTS = {
    'significance_level': 0.05,
    'min_observations': 30,
}

# ============================================================================
# CONFIGURACIÓN CAPM
# ============================================================================
CAPM_DEFAULTS = {
    'max_beta': 2.0,
    'min_observations': 30,
}

# ============================================================================
# CONFIGURACIÓN DE GRÁFICOS Y VISUALIZACIONES
# ============================================================================
PLOTTING_DEFAULTS = {
    'frontier_points': 60,
    'cml_points': 100,
    'sml_points': 100,
    'distribution_bins': 30,
    'report_width': 60,
}

PLOTTING_STYLE = {
    'grid_alpha': 0.3,
    'fill_alpha': 0.3,
    'scatter_size': 100,
    'hspace': 0.3,
    'wspace': 0.3,
}

# ============================================================================
# CONFIGURACIÓN DE OPTIMIZACIÓN
# ============================================================================
OPTIMIZATION_DEFAULTS = {
    'method': 'SLSQP',
    'max_iterations': 1000,
    'tolerance': 1e-6,
}

# ============================================================================
# CONFIGURACIÓN NUMÉRICA
# ============================================================================
NUMERICAL_DEFAULTS = {
    'epsilon': 1e-12,
    'tolerance': 1e-6,
    'weight_tolerance': 1e-6,
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - RATIOS DE RENDIMIENTO
# ============================================================================
RATIO_INTERPRETATION = {
    'sharpe': {
        'excellent': 2.0,
        'very_good': 1.0,
        'acceptable': 0.5,
        'poor': 0.0
    },
    'sortino': {
        'excellent': 2.0,
        'good': 1.0,
        'acceptable': 0.5,
        'poor': 0.0
    },
    'calmar': {
        'excellent': 1.0,
        'good': 0.5,
        'poor': 0.0
    }
}

DRAWDOWN_RISK_LEVELS = {
    'low': 10,      # < 10%
    'moderate': 20, # < 20%
    'high': 30,     # < 30%
    # >= 30% = very_high
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - ESTADÍSTICA
# ============================================================================
INTERPRETATION_THRESHOLDS = {
    'skewness': {
        'positive': 0.5,
        'negative': -0.5
    },
    'kurtosis': {
        'positive': 0
    }
}

# ============================================================================
# UMBRALES DE VALORACIÓN
# ============================================================================
VALUATION_THRESHOLDS = {
    'profitability': {
        'roic': {'excellent': 0.20, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
        'roe': {'excellent': 0.25, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
        'roa': {'excellent': 0.15, 'good': 0.10, 'fair': 0.05, 'poor': 0.02},
        'gross_margin': {'excellent': 0.50, 'good': 0.35, 'fair': 0.20, 'poor': 0.10},
        'operating_margin': {'excellent': 0.25, 'good': 0.15, 'fair': 0.10, 'poor': 0.05},
        'net_margin': {'excellent': 0.20, 'good': 0.10, 'fair': 0.05, 'poor': 0.02},
    },
    'financial_health': {
        'debt_ebitda': {'excellent': 1.0, 'good': 2.0, 'fair': 3.0, 'poor': 5.0},
        'debt_equity': {'excellent': 0.3, 'good': 0.5, 'fair': 1.0, 'poor': 2.0},
        'current_ratio': {'excellent': 2.5, 'good': 2.0, 'fair': 1.5, 'poor': 1.0},
        'interest_coverage': {'excellent': 10.0, 'good': 5.0, 'fair': 3.0, 'poor': 1.5},
    },
    'valuation_multiples': {
        'pb_ratio': {'cheap': 1.5, 'fair': 3.0, 'expensive': 5.0, 'very_expensive': 8.0},
    },
    'efficiency': {
        'asset_turnover': {'excellent': 1.5, 'good': 1.0, 'fair': 0.7, 'poor': 0.4},
        'inventory_turnover': {'excellent': 12.0, 'good': 8.0, 'fair': 5.0, 'poor': 3.0},
        'dso': {'excellent': 30, 'good': 45, 'fair': 60, 'poor': 90},
        'dio': {'excellent': 30, 'good': 60, 'fair': 90, 'poor': 120},
    }
}

# Pesos para scoring de valoración
SCORING_WEIGHTS = {
    'profitability': {
        'roic': 0.30,
        'roe': 0.20,
        'operating_margin': 0.25,
        'net_margin': 0.25,
    },
    'financial_health': {
        'debt_ebitda': 0.25,
        'debt_equity': 0.20,
        'current_ratio': 0.20,
        'net_cash_ebitda': 0.20,
        'free_cash_flow': 0.15,
    }
}

# Umbrales para alertas
ALERT_THRESHOLDS = {
    'profitability': {
        'roic_low': 0.08,
        'operating_margin_low': 0.05,
    },
    'financial_health': {
        'debt_ebitda_danger': 4,
        'debt_ebitda_warning': 3,
        'current_ratio_low': 1.0,
    }
}

# ============================================================================
# CONSTANTES FISCALES Y FINANCIERAS
# ============================================================================
FINANCIAL_CONSTANTS = {
    'default_tax_rate': 0.21,  # 21% tasa impositiva por defecto
}

# ============================================================================
# CONSTANTES DE CONVERSIÓN
# ============================================================================
CONVERSION_FACTORS = {
    'decimal_to_percent': 100,
}

# ============================================================================
# CONSTANTES DE ACCESO RÁPIDO
# ============================================================================
# Factores de anualización
ANNUAL_FACTOR = RISK_ANALYSIS['annual_factor']

# Niveles de confianza
DEFAULT_CONFIDENCE_LEVEL = RISK_ANALYSIS['default_confidence_level']
DEFAULT_CONFIDENCE_LEVELS = RISK_ANALYSIS['default_confidence_levels']

# Monte Carlo
MONTE_CARLO_SIMULATIONS = RISK_ANALYSIS['monte_carlo']['n_simulations']
MONTE_CARLO_SEED = RISK_ANALYSIS['monte_carlo']['seed']

# Rolling windows
ROLLING_WINDOW = RISK_ANALYSIS['rolling']['default_window']

# Estadística
SIGNIFICANCE_LEVEL = STATISTICAL_DEFAULTS['significance_level']
MIN_OBSERVATIONS = STATISTICAL_DEFAULTS['min_observations']

# CAPM
MAX_BETA = CAPM_DEFAULTS['max_beta']

# Gráficos
FRONTIER_POINTS = PLOTTING_DEFAULTS['frontier_points']
CML_POINTS = PLOTTING_DEFAULTS['cml_points']
SML_POINTS = PLOTTING_DEFAULTS['sml_points']
DISTRIBUTION_BINS = PLOTTING_DEFAULTS['distribution_bins']
REPORT_WIDTH = PLOTTING_DEFAULTS['report_width']

# Numérico
EPSILON = NUMERICAL_DEFAULTS['epsilon']
TOLERANCE = NUMERICAL_DEFAULTS['tolerance']

# Optimización
OPTIMIZATION_METHOD = OPTIMIZATION_DEFAULTS['method']

# Columnas yfinance
ADJ_CLOSE_COL = YFINANCE_COLUMNS['adj_close']
CLOSE_COL = YFINANCE_COLUMNS['close']