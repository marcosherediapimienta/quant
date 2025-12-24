"""
CONFIGURACIÓN - ANÁLISIS DE PORTAFOLIO (PM)
Benchmarks, análisis de riesgo, CAPM y valoración.
"""

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
# CONFIGURACIÓN GLOBAL DE FECHAS PARA ANÁLISIS
# ============================================================================
ANALYSIS_DATES = {
    'start_date': '2020-01-01',       # Fecha inicial por defecto para todos los análisis
    'end_date': '2025-12-31',         # Fecha final por defecto
    'use_current_date_as_end': True,  # Si True, ignora end_date y usa fecha actual
    'default_lookback_years': 5,      # Solo se usa si start_date está vacío
}

# ============================================================================
# CONFIGURACIÓN DE DESCARGA (yfinance)
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
    'default_confidence_levels': (0.95, 0.99),
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
# PARÁMETROS ESTADÍSTICOS
# ============================================================================
STATISTICAL_DEFAULTS = {
    'significance_level': 0.05,
    'min_observations': 30,
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
# CONFIGURACIÓN DE OPTIMIZACIÓN
# ============================================================================
OPTIMIZATION_DEFAULTS = {
    'method': 'SLSQP',
    'max_iterations': 1000,
    'tolerance': 1e-6,
    'frontier_points': 60,
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - RATIOS
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
    'low': 10,
    'moderate': 20,
    'high': 30,
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

# ============================================================================
# PESOS PARA SCORING DE VALORACIÓN
# ============================================================================
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

# ============================================================================
# UMBRALES DE ALERTAS
# ============================================================================
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
# CONFIGURACIÓN DE PORTFOLIO
# ============================================================================
PORTFOLIO_CONFIG = {
    'selection': {
        'min_score': 60.0,
        'max_companies': 10,
        'max_per_sector': 3,
        'default_method': 'total_score',
    },
    'scoring_weights': {
        'balanced': {
            'profitability': 0.25,
            'health': 0.25,
            'growth': 0.20,
            'valuation': 0.30
        },
        'value': {
            'total': 0.5,
            'valuation': 0.5
        },
        'growth': {
            'total': 0.5,
            'growth': 0.5
        }
    },
    'optimization': {
        'default_method': 'equal',
        'risk_free_rate': 0.045,
        'annual_trading_days': 252,
        'min_data_points': 30,
        'scipy_method': 'SLSQP',
    },
    'dates': {
        'default_lookback_years': ANALYSIS_DATES['default_lookback_years'],
        'date_format': '%Y-%m-%d',
        'start_date': ANALYSIS_DATES['start_date'],
        'end_date': ANALYSIS_DATES['end_date'],
        'use_current_date_as_end': ANALYSIS_DATES['use_current_date_as_end']
    },
    'defaults': {
        'sector_name': 'Unknown',
        'price_column': 'Close'
    }
}

# ============================================================================
# CONFIGURACIÓN DE SEÑALES DE TRADING (BUY/SELL)
# ============================================================================
TRADING_SIGNALS_CONFIG = {
    'start_date': ANALYSIS_DATES['start_date'],
    'end_date': ANALYSIS_DATES['end_date'],
    'use_current_date_as_end': ANALYSIS_DATES['use_current_date_as_end'],
    'default_lookback_years': ANALYSIS_DATES['default_lookback_years'],
}

# ============================================================================
# CONFIGURACIÓN DE ÍNDICES Y ETFs
# ============================================================================
INDEX_CONFIG = {
    'urls': {
        'sp500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'nasdaq100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'dow30': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
    },
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'etf_mapping': {
        'SPY': 'SP500',
        'VOO': 'SP500',
        'IVV': 'SP500',
        'QQQ': 'NASDAQ100',
        'DIA': 'DOW30'
    },
    'supported_indices': ['SP500', 'NASDAQ100', 'DOW30', 'RUSSELL1000'],
    'validation': {
        'nasdaq_min_tickers': 50,
        'dow_min_tickers': 20,
        'dow_max_tickers': 35,
        'sp500_min_tickers': 400,
    },
    'fallback': {
        'sp500_top100': [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
            'UNH', 'JNJ', 'XOM', 'V', 'JPM', 'WMT', 'PG', 'MA', 'HD', 'CVX',
            'LLY', 'ABBV', 'MRK', 'AVGO', 'KO', 'PEP', 'COST', 'ADBE', 'MCD',
            'CSCO', 'ACN', 'TMO', 'ABT', 'NFLX', 'DHR', 'VZ', 'NKE', 'WFC',
            'DIS', 'CMCSA', 'TXN', 'PM', 'CRM', 'BMY', 'NEE', 'RTX', 'ORCL',
            'INTC', 'UPS', 'HON', 'QCOM', 'BA', 'LOW', 'SPGI', 'BLK', 'LMT',
            'AMD', 'INTU', 'CAT', 'DE', 'GE', 'AXP', 'ISRG', 'GILD', 'NOW',
            'BKNG', 'ADI', 'PLD', 'MDLZ', 'SYK', 'ADP', 'REGN', 'TJX', 'VRTX',
            'CB', 'SBUX', 'CI', 'TMUS', 'PYPL', 'MMC', 'SO', 'ZTS', 'SCHW',
            'MO', 'BSX', 'DUK', 'AMT', 'PGR', 'LRCX', 'EOG', 'ITW', 'BDX',
            'C', 'SLB', 'NOC', 'CME', 'MMM', 'USB', 'HUM', 'PNC', 'FI', 'TGT'
        ],
        'russell_additional': [
            'SNOW', 'CRWD', 'ZS', 'DDOG', 'NET', 'OKTA', 'FTNT',
            'IONQ', 'SMR', 'NIO', 'RIVN', 'LCID'
        ]
    }
}

# ============================================================================
# CONSTANTES DE ACCESO RÁPIDO (para imports directos)
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

# Numérico
EPSILON = NUMERICAL_DEFAULTS['epsilon']
TOLERANCE = NUMERICAL_DEFAULTS['tolerance']

# Optimización
OPTIMIZATION_METHOD = OPTIMIZATION_DEFAULTS['method']
FRONTIER_POINTS = OPTIMIZATION_DEFAULTS['frontier_points']

# Columnas yfinance
ADJ_CLOSE_COL = YFINANCE_COLUMNS['adj_close']
CLOSE_COL = YFINANCE_COLUMNS['close']

# Portfolio
PORTFOLIO_MIN_SCORE = PORTFOLIO_CONFIG['selection']['min_score']
PORTFOLIO_MAX_COMPANIES = PORTFOLIO_CONFIG['selection']['max_companies']
PORTFOLIO_LOOKBACK_YEARS = PORTFOLIO_CONFIG['dates']['default_lookback_years']
PORTFOLIO_RISK_FREE_RATE = PORTFOLIO_CONFIG['optimization']['risk_free_rate']

# Fechas de análisis (⭐ NUEVAS CONSTANTES)
ANALYSIS_START_DATE = ANALYSIS_DATES['start_date']
ANALYSIS_END_DATE = ANALYSIS_DATES['end_date']
USE_CURRENT_DATE = ANALYSIS_DATES['use_current_date_as_end']
ANALYSIS_LOOKBACK_YEARS = ANALYSIS_DATES['default_lookback_years']