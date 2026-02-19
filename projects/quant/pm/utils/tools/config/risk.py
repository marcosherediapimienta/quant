RISK_ANALYSIS = {
    'annual_factor': 252,             
    'risk_free_rate': 0.045,          
    'default_confidence_levels': (0.95, 0.99),
    'default_confidence_level': 0.95,
    'multi_level_confidence': (0.90, 0.95, 0.99),
    'default_var_methods': ['historical', 'parametric', 'monte_carlo'],
    'monte_carlo': {
        'n_simulations': 10000,
        'seed': 42
    },
    'rolling': {
        'default_window': 252
    }
}

STATISTICAL_DEFAULTS = {
    'significance_level': 0.05,
    'min_observations': 30,
}

NUMERICAL_DEFAULTS = {
    'epsilon': 1e-12,
    'tolerance': 1e-6,
    'weight_tolerance': 1e-6,
    'eigenvalue_floor': 1e-10,
}

NORMALITY_THRESHOLDS = {
    'skewness_limit': 0.5,
    'kurtosis_limit': 1.0,
    'score_normal': 3,
    'score_questionable': 2,
}

ANDERSON_DARLING = {
    'critical_value_5pct': 0.787,
    'severity_moderate': 2.0,
    'severity_high': 5.0,
}

TRACKING_ERROR_DEFAULTS = {
    'min_rolling_window': 63,
    'rolling_window_divisor': 4,
}

OPTIMIZATION_DEFAULTS = {
    'method': 'SLSQP',
    'max_iterations': 1000,
    'tolerance': 1e-6,
    'frontier_points': 60,
}

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

SKEWNESS_THRESHOLDS = {
    'positive': 0.5,
    'negative': -0.5
}

KURTOSIS_THRESHOLDS = {
    'leptokurtic': 3,
    'platykurtic': -1
}

SORTINO_THRESHOLDS = {
    'excellent': 2.0,
    'good': 1.0,
    'acceptable': 0.5
}

TRACKING_ERROR_THRESHOLDS = {
    'very_close': 2,
    'moderate': 5,
    'active': 10
}

INFORMATION_RATIO_THRESHOLDS = {
    'excellent': 0.5,
    'positive': 0,
    'slightly_below': -0.5
}

BETA_THRESHOLDS = {
    'aggressive': 1.2,
    'market': 0.8
}

ALPHA_THRESHOLDS = {
    'excellent': 5,
    'positive': 0,
    'slightly_below': -5
}

VAR_RISK_LEVELS = {
    'low': 2,
    'moderate': 5,
    'high': 10
}

CORRELATION_REPORT = {
    'top_n_pairs': 3,
}
