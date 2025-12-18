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

# Divisas por benchmark
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
# DIVISAS Y REGLAS DE INFERENCIA
# ============================================================================
CURRENCY_RULES = {
    'EUR': ['.MC', '.PA', '.DE', '.MI', '.AS', '.BE', '.BR', '.LS', '.F', '-EUR'],
    'GBP': ['.L'],
    'JPY': ['.T'],
    'CAD': ['.TO'],
    'AUD': ['.AX'],
    'CHF': ['.SW'],
    'HKD': ['.HK']
}

DEFAULT_CURRENCY = 'USD'

# ============================================================================
# TIPOS DE ACTIVOS
# ============================================================================
ASSET_CLASS_KEYWORDS = {
    'crypto': ['BTC', 'ETH', 'CRYPTO', 'COIN'],
    'commodity': ['GLD', 'IAU', 'SLV', 'USO', 'UNG', 'DBC', 'GOLD', 'SILVER'],
    'bond': ['TLT', 'IEF', 'SHY', 'AGG', 'BND', 'LQD', 'HYG', 'BOND'],
    'real_estate': ['VNQ', 'IYR', 'REIT'],
    'equity': [] 
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

# ============================================================================
# CONFIGURACIÓN DE RETORNOS
# ============================================================================
RETURN_METHODS = ['log', 'simple']
DEFAULT_RETURN_METHOD = 'log'

# ============================================================================
# FRECUENCIAS TEMPORALES
# ============================================================================
FREQUENCIES = {
    'daily': 'D',
    'weekly': 'W',
    'monthly': 'M',
    'quarterly': 'Q',
    'yearly': 'Y'
}

# ============================================================================
# TIPOS DE PRECIOS
# ============================================================================
PRICE_TYPES = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
DEFAULT_PRICE_TYPE = 'Adj Close'

# ============================================================================
# ANUALIZACIÓN
# ============================================================================
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_WEEKS_PER_YEAR = 52

# ============================================================================
# MÉTODOS DE AGREGACIÓN
# ============================================================================
AGGREGATION_METHODS = ['last', 'first', 'mean', 'sum', 'min', 'max']

# ============================================================================
# MÉTODOS DE RELLENO DE DATOS FALTANTES
# ============================================================================
FILL_METHODS = ['ffill', 'bfill', 'linear', 'nearest']

# ============================================================================
# PARÁMETROS DE CACHÉ
# ============================================================================
DEFAULT_CACHE_HOURS = 24
CACHE_DIR_NAME = 'cache'

# ============================================================================
# RANGOS DE VALIDACIÓN
# ============================================================================
VALIDATION_RANGES = {
    'weight': (0.0, 1.0),
    'correlation': (-1.0, 1.0),
    'beta': (-5.0, 5.0),
    'sharpe': (-10.0, 10.0)
}