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

# Divisas por benchmark (necesario para benchmark_loader)
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
# CONFIGURACIÓN DE DESCARGA (necesario para data_loader)
# ============================================================================
DOWNLOAD_DEFAULTS = {
    'auto_adjust': True,
    'group_by': 'ticker',
    'threads': True,
    'progress': True
}

# ============================================================================
# CONFIGURACIÓN DE ANÁLISIS DE RIESGO
# ============================================================================
RISK_ANALYSIS = {
    'annual_factor': 252,
    'risk_free_rate': 0.045,
    'default_confidence_levels': (0.90, 0.95, 0.99),
    'default_confidence_level': 0.95,
    'monte_carlo': {'n_simulations': 10000,'seed': 42},
    'rolling': {'default_window': 252}
}

# ============================================================================
# UMBRALES DE INTERPRETACIÓN ESTADÍSTICA
# ============================================================================
INTERPRETATION_THRESHOLDS = {
    'skewness': {
        'positive': 0.5,
        'negative': -0.5}, 
    'kurtosis': {'positive': 0}
}