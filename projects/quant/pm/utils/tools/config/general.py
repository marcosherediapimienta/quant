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

ANALYSIS_DATES = {
    'start_date': '2020-01-01',
    'end_date': '2025-12-24',
    'use_current_date_as_end': True,
    'default_lookback_years': 5,
}

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
