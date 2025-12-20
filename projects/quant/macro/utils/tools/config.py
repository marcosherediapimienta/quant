MACRO_FACTORS = {
    # ========== VOLATILIDAD ==========
    'VIX': '^VIX',              # Volatilidad implícita S&P 500
    'VXN': '^VXN',              # Volatilidad implícita NASDAQ
    
    # ========== TIPOS DE INTERÉS USA ==========
    'RATE_3M': '^IRX',          # Tasa 3 meses (13-week T-bill)
    'RATE_2Y': '^IRX',          # Tasa 2 años
    'RATE_5Y': '^FVX',          # Tasa 5 años
    'RATE_10Y': '^TNX',         # Tasa 10 años
    'RATE_30Y': '^TYX',         # Tasa 30 años
    
    # ========== BONOS USA (ETFs) ==========
    'GOVT_1_3Y': 'SHY',         # Treasury 1-3 años
    'GOVT_7_10Y': 'IEF',        # Treasury 7-10 años
    'GOVT_20Y': 'TLT',          # Treasury 20+ años
    'TIPS': 'TIP',              # Treasury Inflation-Protected
    
    # ========== BONOS INTERNACIONALES ==========
    'JPN_BOND': 'JPGB.L',       # Bonos Japón
    'EUR_BOND': 'IEAG.L',       # Bonos Europa agregados
    'GER_BOND': 'DBEU',         # Bonos Alemania
    'UK_BOND': 'IGLT.L',        # Gilts UK
    'EM_BOND': 'EMB',           # Bonos mercados emergentes
    
    # ========== CREDIT SPREADS ==========
    'HYG': 'HYG',               # High yield corporativo
    'LQD': 'LQD',               # Investment grade corporativo
    'JNK': 'JNK',               # High yield alternativo
    
    # ========== DIVISAS ==========
    'DXY': 'DX-Y.NYB',          # Índice dólar
    'EUR_USD': 'EURUSD=X',      # Euro/Dólar
    'GBP_USD': 'GBPUSD=X',      # Libra/Dólar
    'USD_JPY': 'JPY=X',         # Dólar/Yen
    
    # ========== METALES PRECIOSOS ==========
    'GOLD': 'GC=F',             # Oro (futuros)
    'SILVER': 'SI=F',           # Plata (futuros)
    'COPPER': 'HG=F',           # Cobre (futuros)
    
    # ========== ENERGÍA ==========
    'OIL': 'CL=F',              # Petróleo WTI (futuros)
    'BRENT': 'BZ=F',            # Petróleo Brent (futuros)
    'NATGAS': 'NG=F',           # Gas natural (futuros)
    
    # ========== AGRICULTURA ==========
    'WHEAT': 'ZW=F',            # Trigo (futuros)
    'CORN': 'CORN',             # Maíz (futuros)
    
    # ========== ÍNDICES USA ==========
    'SP500': '^GSPC',           # S&P 500
    'NASDAQ': '^IXIC',          # NASDAQ
    'RUSSELL2000': '^RUT',      # Russell 2000 (small cap)
    
    # ========== ÍNDICES INTERNACIONALES ==========
    'DAX': '^GDAXI',            # Alemania
    'FTSE': '^FTSE',            # UK
    'NIKKEI': '^N225',          # Japón
    'HANG_SENG': '^HSI',        # Hong Kong
    'SHANGHAI': '000001.SS',    # China
    'MSCI_EM': 'EEM',           # Mercados emergentes (ETF)
}

# Categorización de factores macro
MACRO_FACTOR_CATEGORIES = {
    'volatility': ['VIX', 'VXN'],
    'interest_rates': ['RATE_3M', 'RATE_2Y', 'RATE_5Y', 'RATE_10Y', 'RATE_30Y'],
    'us_bonds': ['GOVT_1_3Y', 'GOVT_7_10Y', 'GOVT_20Y', 'TIPS'],
    'intl_bonds': ['JPN_BOND', 'EUR_BOND', 'GER_BOND', 'UK_BOND', 'EM_BOND'],
    'credit': ['HYG', 'LQD', 'JNK'],
    'currencies': ['DXY', 'EUR_USD', 'GBP_USD', 'USD_JPY'],
    'metals': ['GOLD', 'SILVER', 'COPPER'],
    'energy': ['OIL', 'BRENT', 'NATGAS'],
    'agriculture': ['WHEAT', 'CORN'],
    'us_indices': ['SP500', 'NASDAQ', 'RUSSELL2000'],
    'intl_indices': ['DAX', 'FTSE', 'NIKKEI', 'HANG_SENG', 'SHANGHAI', 'MSCI_EM']
}

# Factores macro core (esenciales para análisis básico)
MACRO_CORE_FACTORS = [
    'VIX',           # Volatilidad
    'RATE_10Y',      # Tipo de interés largo
    'RATE_3M',       # Tipo de interés corto
    'DXY',           # Dólar
    'GOLD',          # Refugio/inflación
    'OIL'            # Energía/crecimiento
]

# Factores para análisis de situación macro global
MACRO_GLOBAL_FACTORS = [
    # Curva de tipos USA completa
    'RATE_2Y', 'RATE_5Y', 'RATE_10Y', 'RATE_30Y',
    # Bonos globales
    'GOVT_20Y', 'JPN_BOND', 'EUR_BOND', 'GER_BOND',
    # Credit
    'HYG', 'LQD',
    # Inflación proxies
    'GOLD', 'OIL', 'COPPER', 'SILVER',
    # Volatilidad
    'VIX',
    # Divisas
    'DXY', 'EUR_USD',
    # Índices principales
    'SP500', 'NIKKEI', 'DAX'
]

# Categorización de factores macro
MACRO_FACTOR_CATEGORIES = {
    'volatility': ['VIX', 'VXN'],
    'interest_rates': ['RATE_3M', 'RATE_10Y', 'RATE_30Y'],
    'currencies': ['DXY', 'EUR_USD', 'GBP_USD', 'USD_JPY'],
    'commodities': ['GOLD', 'OIL', 'COPPER'],
    'bonds': ['TLT', 'IEF', 'SHY'],
    'credit': ['HYG', 'LQD'],
    'equity_indices': ['SP500', 'NASDAQ', 'RUSSELL2000']
}

# Factores macro core (esenciales para análisis básico)
MACRO_CORE_FACTORS = [
    'VIX',           # Volatilidad
    'RATE_10Y',      # Tipo de interés largo
    'RATE_3M',       # Tipo de interés corto
    'DXY',           # Dólar
    'GOLD',          # Refugio
    'OIL'            # Energía/crecimiento
]

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS MACRO
# ============================================================================
MACRO_ANALYSIS = {
    # Correlaciones
    'correlation': {
        'default_lag_window': 126,        # ~6 meses hábiles
        'min_observations': 60,            # Mínimo de observaciones
        'max_lag': 126,                    # Lags máximos a probar
        'hac_maxlags': 5,                  # Lags para Newey-West
    },
    
    # Regresión multifactor
    'regression': {
        'min_observations': 100,           # Mínimo para regresión
        'hac_maxlags': None,              # None = auto (sqrt(n))
        'significance_level': 0.05,        # Nivel de significancia
    },
    
    # Regímenes
    'regime': {
        'window': 252,                    # Ventana para clustering
        'step': 21,                       # Paso walk-forward
        'n_clusters': 3,                 # Número de clusters
        'features': ['vol', 'ret', 'sharpe', 'mdd'],  # Features para clustering
    },
    
    # Señales
    'signal': {
        'percentile_window': 252,         # Ventana para percentiles
        'upper_threshold': 0.8,           # Umbral superior (risk-off)
        'lower_threshold': 0.2,           # Umbral inferior (risk-on)
        'neutral_range': (0.2, 0.8),      # Rango neutral
    },
    
    # Transformaciones
    'transforms': {
        'yield_scale': 10.0,              # Yahoo escala yields x10
        'use_log_returns': True,          # Usar log returns vs pct_change
    },
}

# ============================================================================
# TRANSFORMACIONES DE FACTORES MACRO
# ============================================================================
MACRO_TRANSFORMS = {
    # Factores que requieren división por 10 (Yahoo escala)
    'yield_factors': ['RATE_3M', 'RATE_10Y', 'RATE_30Y'],
    
    # Factores que usan diferencias (no retornos)
    'diff_factors': ['RATE_3M', 'RATE_10Y', 'RATE_30Y',
                     'VIX'],
    
    # Factores que usan log returns
    'log_return_factors': [
        'DXY', 'EUR_USD', 'GBP_USD', 'USD_JPY',
        'GOLD', 'OIL', 'COPPER',
        'SP500', 'NASDAQ', 'RUSSELL2000',
        'TLT', 'IEF', 'SHY', 'HYG', 'LQD',
        'GOVT_20Y'
    ],
    
    # Factores que se usan directamente (niveles)
    'level_factors': ['VIX', 'VXN'],
}

# ============================================================================
# SPREADS Y COMBINACIONES DE FACTORES
# ============================================================================
MACRO_SPREADS = {
    # Spread de curva de tipos (expectativas económicas)
    'yield_curve_30_10y': {
        'long': 'RATE_30Y',
        'short': 'RATE_10Y',
        'transform': 'diff',
    },
    
    # Credit spread amplio (riesgo crediticio vs riesgo libre)
    'credit_spread_hy': {
        'risky': 'HYG',
        'safe': 'GOVT_20Y',
        'transform': 'diff',
    },
    
    # Credit spread fino (high yield vs investment grade)
    'credit_spread_hy_lqd': {
        'risky': 'HYG',
        'safe': 'LQD',
        'transform': 'diff',
    },
}

FACTORS_TO_USE = [
    # Volatilidad (riesgo de mercado)
    'VIX',
    
    # Política monetaria (descuento de flujos)
    'RATE_2Y',   # Expectativas corto plazo
    'RATE_10Y',  # Tipo largo plazo
    'RATE_30Y',  # Tipo muy largo plazo (para spread 30-10Y)
    
    # Bonos (para credit spread)
    'GOVT_20Y',  # TLT - Treasury 20+ años (para credit spread)
    
    # Credit spreads (apetito por riesgo)
    'HYG',       # High yield
    'LQD',       # Investment grade
    
    # Dólar (afecta exportaciones tech)
    'DXY',
    
    # Commodities (inflación/crecimiento)
    'GOLD',      # Refugio seguro
    'OIL',       # Energía/crecimiento
    
    # Equity market (correlación)
    'SP500'
]

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - SEÑALES MACRO
# ============================================================================
MACRO_SIGNAL_THRESHOLDS = {
    'vix': {
        'low': 15,           # Risk-on
        'moderate': 20,
        'high': 25,          # Risk-off
        'extreme': 30,       # Pánico
    },
    'yield_curve': {
        'inverted': -0.5,    # Spread negativo (recesión)
        'flat': 0.5,          # Spread pequeño
        'normal': 1.5,       # Spread normal
        'steep': 2.5,        # Curva empinada
    },
    'dxy': {
        'weak': 95,          # Dólar débil
        'neutral': 100,
        'strong': 105,       # Dólar fuerte
        'very_strong': 110,
    },
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - CORRELACIONES
# ============================================================================
MACRO_CORRELATION_THRESHOLDS = {
    'very_strong': 0.8,
    'strong': 0.6,
    'moderate': 0.4,
    'weak': 0.2,
    'negligible': 0.0,
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN - FACTOR LOADINGS
# ============================================================================
MACRO_LOADING_THRESHOLDS = {
    'very_high': 1.5,
    'high': 1.0,
    'moderate': 0.5,
    'low': 0.2,
    'negligible': 0.0,
}

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIONES MACRO
# ============================================================================
MACRO_PLOTTING = {
    'dashboard_figsize': (22, 16),
    'correlation_heatmap_figsize': (12, 10),
    'factor_loading_figsize': (14, 8),
    'regime_figsize': (16, 10),
    'signal_figsize': (16, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8',
}

# ============================================================================
# CONSTANTES DE ACCESO RÁPIDO
# ============================================================================
# Correlaciones
DEFAULT_LAG_WINDOW = MACRO_ANALYSIS['correlation']['default_lag_window']
MAX_LAG = MACRO_ANALYSIS['correlation']['max_lag']
CORRELATION_MIN_OBS = MACRO_ANALYSIS['correlation']['min_observations']
HAC_MAXLAGS = MACRO_ANALYSIS['correlation']['hac_maxlags']

# Regresión
REGRESSION_MIN_OBS = MACRO_ANALYSIS['regression']['min_observations']
REGRESSION_SIGNIFICANCE = MACRO_ANALYSIS['regression']['significance_level']

# Regímenes
REGIME_WINDOW = MACRO_ANALYSIS['regime']['window']
REGIME_STEP = MACRO_ANALYSIS['regime']['step']
REGIME_N_CLUSTERS = MACRO_ANALYSIS['regime']['n_clusters']

# Señales
SIGNAL_PERCENTILE_WINDOW = MACRO_ANALYSIS['signal']['percentile_window']
SIGNAL_UPPER_THRESHOLD = MACRO_ANALYSIS['signal']['upper_threshold']
SIGNAL_LOWER_THRESHOLD = MACRO_ANALYSIS['signal']['lower_threshold']

# Transformaciones
YIELD_SCALE = MACRO_ANALYSIS['transforms']['yield_scale']
USE_LOG_RETURNS = MACRO_ANALYSIS['transforms']['use_log_returns']