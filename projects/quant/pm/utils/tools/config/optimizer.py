OPTIMIZER_NUMERICAL = {
    'vol_floor': 1e-10,
    'penalty': 1e10,
    'cov_regularization': 1e-8,
    'max_shrinkage': 0.1,
    'risk_parity_min_bound': 1e-4,
    'default_seed': 42,
    'default_maxiter': 1000,
    'default_ftol': 1e-9,
    'risk_parity_seed': 0,
    'risk_parity_maxiter': 2000,
    'risk_parity_ftol': 1e-12,
}

OPTIMIZER_BLACK_LITTERMAN = {
    'delta_min': 2.0,
    'delta_max': 4.0,
    'tau_min': 0.01,
    'tau_max': 0.1,
    'neutral_score': 50.0,
    'max_view_return': 0.15,
    'min_confidence': 0.1,
    'cond_threshold': 1e12,
}
