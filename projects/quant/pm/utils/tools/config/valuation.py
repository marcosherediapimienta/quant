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

SCORING_RANGES = {
    'growth': {
        'revenue': {'min': -0.20, 'max': 0.40},
        'earnings': {'min': -0.30, 'max': 0.50},
    },
    'profitability': {
        'margin': {'min': -0.10, 'max': 0.50},
    }
}

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

ALERT_THRESHOLDS = {
    'profitability': {
        'roic_low': 0.08,
        'operating_margin_low': 0.05,
    },
    'financial_health': {
        'debt_ebitda_danger': 4,
        'debt_ebitda_warning': 3,
        'current_ratio_low': 1.0,
    },
    'growth': {
        'revenue_decline_significant': -0.10,
        'earnings_decline_strong': -0.20,
        'growth_too_high': 0.50,
        'revenue_decline_mild': -0.05,
        'earnings_vs_revenue_multiple': 2,
        'high_earnings_growth_threshold': 0.20
    },
    'efficiency': {
        'dso_high': 60,
        'dio_high': 90,
        'asset_turnover_low': 0.5
    },
    'valuation': {
        'pe_very_high': 40,
        'pe_negative': 0,
        'ev_ebitda_high': 20,
        'peg_high': 2,
        'fcf_yield_negative': 0
    }
}

PROFITABILITY_SCORING_RANGES = {
    'roic': {'min': -0.10, 'max': 0.30},
    'roe': {'min': -0.10, 'max': 0.35},
    'operating_margin': {'min': -0.10, 'max': 0.30},
    'net_margin': {'min': -0.10, 'max': 0.25}
}

FINANCIAL_HEALTH_SCORING = {
    'weights': {
        'debt_ebitda': 0.25,
        'debt_equity': 0.20,
        'current_ratio': 0.20,
        'net_cash_ebitda': 0.20,
        'free_cash_flow': 0.15
    },
    'ranges': {
        'debt_ebitda': {'min': 0, 'max': 6},
        'debt_equity': {'min': 0, 'max': 3},
        'current_ratio': {'min': 0.5, 'max': 3.0},
        'net_cash_ebitda': {'min': -3, 'max': 3}
    }
}

EFFICIENCY_SCORING = {
    'weights': {
        'asset_turnover': 0.40,
        'dso': 0.30,
        'dio': 0.30
    },
    'ranges': {
        'asset_turnover': {'min': 0.2, 'max': 2.0},
        'dso': {'min': 20, 'max': 90},
        'dio': {'min': 20, 'max': 120}
    }
}

VALUATION_SCORING = {
    'weights': {
        'pe_ttm': 0.20,
        'ev_ebitda': 0.20,
        'pb_ratio': 0.15,
        'fcf_yield': 0.25,
        'peg_ratio': 0.20
    },
    'ranges': {
        'pe_ttm': {'min': 5, 'max': 40},
        'ev_ebitda': {'min': 4, 'max': 25},
        'pb_ratio': {'min': 1.5, 'max': 8},
        'fcf_yield': {'min': -0.02, 'max': 0.12},
        'peg_ratio': {'min': 0.5, 'max': 3.5}
    }
}

GROWTH_SCORING_WEIGHTS = {
    'revenue': 0.50,
    'earnings': 0.50
}

TRADING_SIGNAL_RULES = {
    'buy': {
        'strong': {
            'valuation_min': 60,
            'fundamental_min': 80,
            'confidence_base': 75,
            'confidence_max': 95,
            'valuation_weight': 0.4,
            'fundamental_weight': 0.3
        },
        'moderate': {
            'valuation_min': 60,
            'fundamental_min': 70,
            'confidence_base': 65,
            'confidence_max': 85,
            'valuation_weight': 0.4,
            'fundamental_weight': 0.2
        },
        'value': {
            'valuation_min': 60,
            'fundamental_min': 60,
            'confidence_base': 55,
            'confidence_max': 75,
            'valuation_weight': 0.4
        },
        'quality': {
            'valuation_min': 45,
            'fundamental_min': 85,
            'confidence_base': 60,
            'confidence_max': 75,
            'fundamental_weight': 0.5
        }
    },
    'sell': {
        'strong': {
            'valuation_max': 20,
            'fundamental_max': 60,
            'confidence_base': 70,
            'confidence_max': 95,
            'valuation_weight': 0.9,
            'fundamental_weight': 0.5
        },
        'moderate': {
            'valuation_max': 40,
            'fundamental_max': 65,
            'confidence_base': 55,
            'confidence_max': 85,
            'valuation_weight': 0.8,
            'fundamental_weight': 0.4
        },
        'overvalued': {
            'valuation_max': 15,
            'confidence_base': 60,
            'confidence_max': 75,
            'valuation_weight': 1.0
        }
    },
    'hold': {
        'mixed_quality_price': {
            'valuation_max': 40,
            'fundamental_min': 85,
            'confidence': 50
        },
        'mixed_moderate': {
            'valuation_max': 45,
            'fundamental_min': 75,
            'confidence': 50
        },
        'default': {
            'confidence': 50
        }
    },
    'sanity_check': {
        'sell_override_to_hold': {
            'upside_min': 0.10,
            'confidence': 50
        },
        'buy_override_to_hold': {
            'upside_max': -0.10,
            'confidence': 50
        }
    }
}

PRICE_TARGET_CONFIG = {
    'peg_method': {
        'fair_peg': 1.0,
        'adjustment_divisor_bear': 100,
        'adjustment_divisor_bull': 200
    },
    'pe_method': {
        'growth_multiplier': 1.5,
        'pe_weight': 0.3,
        'growth_weight': 0.7,
        'fair_multiplier_base': 0.7,
        'fair_multiplier_range': 0.6,
        'adjustment_divisor': 333,
        'earnings_growth_threshold': 2
    },
    'analyst_method': {
        'confidence_base': 0.5,
        'confidence_range': 0.5,
    },
    'score_method': {
        'adjustment_divisor_bear': 250,
        'adjustment_divisor_bull': 333
    }
}

OVERALL_VALUATION_LOGIC = {
    'thresholds': {
        'pe': {
            'cheap': 15,
            'expensive': 25,
            'min_valid': 0,
            'max_valid': 100
        },
        'ev_ebitda': {
            'cheap': 10,
            'expensive': 15,
            'min_valid': 0,
            'max_valid': 100
        },
        'fcf_yield': {
            'cheap': 0.06,
            'expensive': 0.02,
            'min_valid': -0.5,
            'max_valid': 0.5
        },
        'pb': {
            'cheap': 2,
            'expensive': 5,
            'min_valid': 0,
            'max_valid': 50
        }
    },
    'voting': {
        'min_valid_metrics': 2,
        'min_votes_for_decision': 2
    }
}

VALUATION_MULTIPLES_FALLBACKS = {
    'pe_ratio': {
        'fair': 18,
        'very_expensive': 35
    },
    'ev_ebitda': {
        'fair': 12,
        'very_expensive': 20
    },
    'fcf_yield': {
        'good': 0.05,
        'fair': 0.03
    }
}

SECTOR_ANALYSIS_CONFIG = {
    'max_peers': 10,
    'percentile_thresholds': {
        'top_performer': 80,
        'above_average': 60,
        'average': 40,
        'below_average': 20
    },
    'percentile_labels': {
        'top': 'Top performer del sector',
        'above': 'Por encima del promedio',
        'average': 'En el promedio del sector',
        'below': 'Por debajo del promedio',
        'bottom': 'Rezagado del sector'
    }
}

SECTOR_PEERS = {
    'Technology': ['MSFT', 'AAPL', 'GOOGL', 'META', 'AMZN', 'NVDA', 'ORCL', 'ADBE', 'CRM'],
    'Financial Services': ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'BLK'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX'],
    'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ'],
    'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM'],
    'Industrials': ['CAT', 'HON', 'UPS', 'UNP', 'BA', 'GE', 'DE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC'],
    'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM'],
}

REPORTING_CONFIG = {
    'top_opportunities': 5,
    'max_reasons_display': 2,
    'line_width': 80,
    'days_per_year': 365
}

SCORE_INTERPRETATION = {
    'default_na_score': 50,
    'reasons': {
        'valuation': {
            'excellent': 70,
            'good': 55,
            'poor': 45,
            'very_poor': 30
        },
        'fundamental': {
            'exceptional': 80,
            'good': 70,
            'weak': 50
        },
        'profitability': {
            'excellent': 80,
            'poor': 40
        },
        'health': {
            'excellent': 80,
            'poor': 40
        },
        'growth': {
            'high': 75,
            'low': 35
        }
    },
    'visual': {
        'excellent': 80,
        'good': 60,
        'fair': 40
    }
}

SCORE_AGGREGATION_WEIGHTS = {
    'total': {
        'valuation': 0.40,
        'fundamental': 0.50,
        'technical': 0.10
    },
    'fundamental': {
        'profitability': 0.35,
        'health': 0.35,
        'growth': 0.30
    }
}

COMPANY_ANALYSIS_WEIGHTS = {
    'default': {
        'profitability': 0.25,
        'financial_health': 0.25,
        'growth': 0.20,
        'efficiency': 0.15,
        'valuation': 0.15
    }
}

INVESTMENT_PROFILES = {
    'balanced': {
        'profitability': 0.30,
        'financial_health': 0.30,
        'growth': 0.15,
        'efficiency': 0.10,
        'valuation': 0.15,
        'description': 'Para inversores que buscan equilibrio'
    },
    'value': {
        'profitability': 0.20,
        'financial_health': 0.25,
        'growth': 0.10,
        'efficiency': 0.10,
        'valuation': 0.35,
        'description': 'Para inversores de valor (bajo P/E, alto dividend yield)'
    },
    'growth': {
        'profitability': 0.15,
        'financial_health': 0.15,
        'growth': 0.45,
        'efficiency': 0.10,
        'valuation': 0.15,
        'description': 'Para inversores de crecimiento (alto revenue growth)'
    },
    'quality': {
        'profitability': 0.35,
        'financial_health': 0.35,
        'growth': 0.15,
        'efficiency': 0.10,
        'valuation': 0.05,
        'description': 'Para inversores que priorizan calidad (alto ROE, bajo debt)'
    }
}

CONCLUSION_THRESHOLDS = {
    'excellent': 80,
    'good': 65,
    'fair': 50,
    'weak': 35,
    'labels': {
        'excellent': 'EXCELLENT',
        'good': 'GOOD',
        'fair': 'FAIR',
        'weak': 'WEAK',
        'critical': 'CRITICAL',
        'insufficient': 'INSUFFICIENT DATA'
    }
}
