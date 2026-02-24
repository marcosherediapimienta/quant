from .general import ANALYSIS_DATES
from .indices import SP500_FALLBACK, IBEX_35, EUROSTOXX_50, NIKKEI_225

PORTFOLIO_CONFIG = {
    'selection': {
        'min_score': 60.0,
        'max_companies': 10,
        'max_per_sector': 3,
        'default_method': 'total_score',
    },
    'scoring_weights': {
        'balanced': {
            'profitability': 0.30,
            'health': 0.30,
            'growth': 0.20,
            'valuation': 0.20
        },
        'value': {
            'total': 0.5,
            'valuation': 0.5
        },
        'growth': {
            'total': 0.5,
            'growth': 0.5
        },
        'quality': {
            'profitability': 0.45,
            'health': 0.40,
            'growth': 0.10,
            'valuation': 0.05
        }
    },
    'selection_thresholds': {
        'balanced': {
            'profitability': 40.0,
            'health': 40.0,
            'growth': 35.0,
            'valuation': 30.0,
        },
        'value': {
            'valuation': 50.0,
            'health': 35.0,
        },
        'growth': {
            'growth': 50.0,
            'health': 30.0,
        },
        'quality': {
            'profitability': 50.0,
            'health': 50.0,
        },
        'total_score': {},
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
        'ibex35': 'https://en.wikipedia.org/wiki/IBEX_35',
        'eurostoxx50': 'https://en.wikipedia.org/wiki/EURO_STOXX_50',
        'nikkei225': 'https://en.wikipedia.org/wiki/Nikkei_225',
    },
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'etf_mapping': {
        'SPY': 'SP500',
        'VOO': 'SP500',
        'IVV': 'SP500',
        'QQQ': 'NASDAQ100',
        'DIA': 'DOW30',
        'EWP': 'IBEX35',
        'FEZ': 'EUROSTOXX50',
        'EWJ': 'NIKKEI225',
        'URTH': 'MSCI_WORLD',
    },
    'supported_indices': ['SP500', 'NASDAQ100', 'DOW30', 'IBEX35', 'EUROSTOXX50', 'NIKKEI225', 'MSCI_WORLD'],
    'validation': {
        'nasdaq_min_tickers': 50,
        'dow_min_tickers': 20,
        'dow_max_tickers': 35,
        'sp500_min_tickers': 400,
        'ibex35_min_tickers': 25,
        'ibex35_max_tickers': 40,
        'eurostoxx50_min_tickers': 50,
        'eurostoxx50_max_tickers': 50,
        'nikkei225_min_tickers': 150,
        'nikkei225_max_tickers': 260,
    },
    'fallback': {
        'sp500': SP500_FALLBACK,
        'ibex35': IBEX_35,
        'eurostoxx50': EUROSTOXX_50,
        'nikkei225': NIKKEI_225,
    }
}