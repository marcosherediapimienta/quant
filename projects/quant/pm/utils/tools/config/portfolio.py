from .general import ANALYSIS_DATES

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
        'default_method': 'score_risk_adjusted',
        'risk_free_rate': 0.045,
        'annual_trading_days': 252,
        'min_data_points': 30,
        'scipy_method': 'SLSQP',
        'max_weight': 1.0,
        'min_weight': 0.0,
        'n_restarts': 5,
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
    },
    'analysis': {
        'quick_analysis_threshold': 50,
        'candidate_multiplier': 3,
        'min_candidates': 50,
        'max_workers_quick': 10,
        'max_workers_full': 3,
        'rate_limit_interval': 0.5,
        'retry_delay_quick': 2,
        'retry_delay_full': 1,
        'log_interval_quick': 50,
        'log_interval_full': 10,
    },
}

TRADING_SIGNALS_CONFIG = {
    'start_date': ANALYSIS_DATES['start_date'],
    'end_date': ANALYSIS_DATES['end_date'],
    'use_current_date_as_end': ANALYSIS_DATES['use_current_date_as_end'],
    'default_lookback_years': ANALYSIS_DATES['default_lookback_years'],
}

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
