MACRO_FACTORS = {
    # Volatilidad
    'VIX': '^VIX',
    'VXN': '^VXN',
    
    # Tipos de interés USA
    'RATE_3M': '^IRX',
    'RATE_2Y': '^IRX',
    'RATE_5Y': '^FVX',
    'RATE_10Y': '^TNX',
    'RATE_30Y': '^TYX',
    
    # Bonos USA (ETFs)
    'GOVT_1_3Y': 'SHY',
    'GOVT_7_10Y': 'IEF',
    'GOVT_20Y': 'TLT',
    'TIPS': 'TIP',
    
    # Bonos internacionales (ETFs)
    'JPN_BOND': 'DBJP',      # iShares Japan Bond ETF
    'EUR_BOND': 'IBND',      # SPDR International Corporate Bond ETF
    'GER_BOND': 'BUNL',      # iShares Germany Government Bond ETF
    'UK_BOND': 'IGOV',       # iShares International Treasury Bond ETF (includes UK)
    'EM_BOND': 'EMB',        # iShares JP Morgan USD Emerging Markets Bond ETF
    'CHINA_BOND': 'CBON',    # VanEck China Bond ETF
    'CAN_BOND': 'AGG',       # iShares Core US Aggregate Bond ETF (proxy for Canada)
    'AUS_BOND': 'BWX',       # SPDR Bloomberg International Treasury Bond ETF (proxy for Australia)
    'INTL_BOND': 'BWX',      # SPDR Bloomberg International Treasury Bond ETF (general international)
    
    # Credit spreads
    'HYG': 'HYG',
    'LQD': 'LQD',
    'JNK': 'JNK',
    
    # Divisas
    'DXY': 'DX-Y.NYB',
    'EUR_USD': 'EURUSD=X',
    'GBP_USD': 'GBPUSD=X',
    'USD_JPY': 'JPY=X',
    
    # Metales preciosos
    'GOLD': 'GC=F',
    'SILVER': 'SI=F',
    'COPPER': 'HG=F',
    
    # Energía
    'OIL': 'CL=F',
    'BRENT': 'BZ=F',
    'NATGAS': 'NG=F',
    
    # Agricultura
    'WHEAT': 'ZW=F',
    'CORN': 'CORN',
    
    # Índices USA
    'SP500': '^GSPC',
    'NASDAQ': '^IXIC',
    'RUSSELL2000': '^RUT',
    
    # Índices internacionales
    'DAX': '^GDAXI',
    'FTSE': '^FTSE',
    'NIKKEI': '^N225',
    'HANG_SENG': '^HSI',
    'SHANGHAI': '000001.SS',
    'MSCI_EM': 'EEM',
}

# ============================================================================
# CATEGORÍAS DE FACTORES (para organizar y seleccionar)
# ============================================================================
MACRO_FACTOR_CATEGORIES = {
    'volatility': ['VIX', 'VXN'],
    'interest_rates': ['RATE_3M', 'RATE_2Y', 'RATE_5Y', 'RATE_10Y', 'RATE_30Y'],
    'us_bonds': ['GOVT_1_3Y', 'GOVT_7_10Y', 'GOVT_20Y', 'TIPS'],
    'intl_bonds': ['JPN_BOND', 'EUR_BOND', 'GER_BOND', 'UK_BOND', 'EM_BOND', 'CHINA_BOND', 'CAN_BOND', 'AUS_BOND', 'INTL_BOND'],
    'credit': ['HYG', 'LQD', 'JNK'],
    'currencies': ['DXY', 'EUR_USD', 'GBP_USD', 'USD_JPY'],
    'metals': ['GOLD', 'SILVER', 'COPPER'],
    'energy': ['OIL', 'BRENT', 'NATGAS'],
    'agriculture': ['WHEAT', 'CORN'],
    'us_indices': ['SP500', 'NASDAQ', 'RUSSELL2000'],
    'intl_indices': ['DAX', 'FTSE', 'NIKKEI', 'HANG_SENG', 'SHANGHAI', 'MSCI_EM']
}

# ============================================================================
# SETS DE FACTORES PREDEFINIDOS (para análisis rápidos)
# ============================================================================

# Factores esenciales (6 factores core)
MACRO_CORE_FACTORS = [
    'VIX',           # Volatilidad
    'RATE_10Y',      # Tipo de interés largo
    'RATE_3M',       # Tipo de interés corto
    'DXY',           # Dólar
    'GOLD',          # Refugio/inflación
    'OIL'            # Energía/crecimiento
]

# Factores para análisis completo (17 factores)
MACRO_GLOBAL_FACTORS = [
    # Curva de tipos USA
    'RATE_2Y', 'RATE_5Y', 'RATE_10Y', 'RATE_30Y',
    # Bonos
    'GOVT_20Y', 'JPN_BOND', 'EUR_BOND', 'GER_BOND',
    # Credit
    'HYG', 'LQD',
    # Commodities (inflación)
    'GOLD', 'OIL', 'COPPER', 'SILVER',
    # Volatilidad
    'VIX',
    # Divisas
    'DXY', 'EUR_USD',
    # Índices
    'SP500', 'NIKKEI', 'DAX'
]

# Factores para portafolios tech (tu caso de uso actual)
FACTORS_TO_USE = [
    'VIX',           # Riesgo de mercado
    'RATE_2Y',       # Expectativas corto plazo
    'RATE_10Y',      # Tipo largo plazo
    'RATE_30Y',      # Para spread 30-10Y
    'GOVT_20Y',      # Treasury para credit spread
    'HYG',           # High yield
    'LQD',           # Investment grade
    'DXY',           # Dólar
    'GOLD',          # Refugio
    'OIL',           # Energía/inflación
    'SP500'          # Mercado
]

# ============================================================================
# TRANSFORMACIONES (cómo procesar cada tipo de factor)
# ============================================================================
MACRO_TRANSFORMS = {
    # Factores que requieren división por 10 (Yahoo escala yields)
    'yield_factors': ['RATE_3M', 'RATE_10Y', 'RATE_30Y', 'RATE_2Y', 'RATE_5Y'],
    
    # Factores que usan diferencias (no retornos)
    'diff_factors': ['RATE_3M', 'RATE_10Y', 'RATE_30Y', 'RATE_2Y', 'RATE_5Y', 'VIX'],
    
    # Factores que usan log returns
    'log_return_factors': [
        'DXY', 'EUR_USD', 'GBP_USD', 'USD_JPY',
        'GOLD', 'OIL', 'COPPER', 'SILVER',
        'SP500', 'NASDAQ', 'RUSSELL2000',
        'TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'GOVT_20Y'
    ],
}

# ============================================================================
# SPREADS (combinaciones de factores)
# ============================================================================
MACRO_SPREADS = {
    # Curva de tipos (expectativas de crecimiento)
    'yield_curve_30_10y': {
        'long': 'RATE_30Y',
        'short': 'RATE_10Y',
        'transform': 'diff',
    },
    
    # Credit spread HY vs Treasury (riesgo crediticio)
    'credit_spread_hy': {
        'risky': 'HYG',
        'safe': 'GOVT_20Y',
        'transform': 'diff',
    },
    
    # Credit spread HY vs IG (apetito por riesgo)
    'credit_spread_hy_lqd': {
        'risky': 'HYG',
        'safe': 'LQD',
        'transform': 'diff',
    },
}

# ============================================================================
# PARÁMETROS DE ANÁLISIS
# ============================================================================
MACRO_ANALYSIS = {
    # Correlaciones
    'correlation': {
        'default_lag_window': 126,    # ~6 meses
        'min_observations': 60,
        'max_lag': 30,                # Reducido de 126 a 30 días (~1 mes)
        'hac_maxlags': 5,
    },
    
    # Regresión multifactor
    'regression': {
        'min_observations': 100,      # Mínimo para regresión robusta
        'hac_maxlags': None,          # None = auto (sqrt(n))
        'significance_level': 0.05,
    },
    
    # Regímenes de mercado
    'regime': {
        'window': 252,
        'step': 21,
        'n_clusters': 3,
        'features': ['vol', 'ret', 'sharpe', 'mdd'],
    },
    
    # Señales
    'signal': {
        'percentile_window': 252,
        'upper_threshold': 0.8,
        'lower_threshold': 0.2,
    },
    
    # Transformaciones
    'transforms': {
        'yield_scale': 10.0,         
        'use_log_returns': True,
    },
}

# ============================================================================
# PARÁMETROS DE ANÁLISIS DE SITUACIÓN MACRO
# ============================================================================
MACRO_SITUATION_THRESHOLDS = {
    # Períodos de tiempo (en días de trading)
    'periods': {
        'week': 5,
        'month': 21,
        'quarter': 63,
        'year': 252,
    },
    
    # VIX levels
    'vix': {
        'panic': 35,
        'stress': 25,
        'tension': 20,
        'normal': 15,
    },
    
    # Yield curve spreads (en puntos porcentuales)
    'yield_curve': {
        'inverted': 0.0,
        'flat': 0.3,
        'steep': 2.0,
    },
    
    # Inflación (cambio % anual en commodities)
    'inflation': {
        'high': 15,
        'moderate': 5,
        'low': -5,
    },
    
    # Tendencias de dólar/oro (cambio %)
    'trends': {
        'strong_move': 5,
        'moderate_move': 3,
        'significant_gold': 10,
        'divergence_threshold': 0.5,
        'momentum_ratio': 0.4,
    },
    
    # Bonos globales (cambio %)
    'bonds': {
        'severe_drop': -10,
        'moderate_drop': -5,
        'strong_gain': 10,
        'moderate_gain': 5,
    },
}

# Lags para análisis de correlación
CORRELATION_LAGS_DEFAULT = [0, 1, 5, 21, 63, 126]  # 0d, 1d, 1w, 1m, 3m, 6m

# Thresholds para sensibilidades (betas)
SENSITIVITY_THRESHOLDS = {
    'high': 0.5,
    'moderate': 0.2,
    'low': 0.0,
}

# ============================================================================
# CONSTANTES DE ACCESO RÁPIDO (para imports directos)
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

# Anualización
ANNUAL_FACTOR = 252

# Constantes de acceso rápido
PERIOD_WEEK = MACRO_SITUATION_THRESHOLDS['periods']['week']
PERIOD_MONTH = MACRO_SITUATION_THRESHOLDS['periods']['month']
PERIOD_QUARTER = MACRO_SITUATION_THRESHOLDS['periods']['quarter']
PERIOD_YEAR = MACRO_SITUATION_THRESHOLDS['periods']['year']
