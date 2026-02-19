import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Optional, Tuple
from scipy.optimize import minimize
from ....tools.config import PORTFOLIO_CONFIG

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# Numerical constants
_VOL_FLOOR = 1e-10
_PENALTY = 1e10
_COV_REGULARIZATION = 1e-8
_MAX_SHRINKAGE = 0.1
_RISK_PARITY_MIN_BOUND = 1e-4

# Black-Litterman constants
_BL_DELTA_MIN = 2.0
_BL_DELTA_MAX = 4.0
_BL_TAU_MIN = 0.01
_BL_TAU_MAX = 0.1
_BL_NEUTRAL_SCORE = 50.0
_BL_MAX_VIEW_RETURN = 0.15
_BL_MIN_CONFIDENCE = 0.1
_BL_COND_THRESHOLD = 1e12


def _shrink_covariance(returns_matrix: np.ndarray, annual_factor: int) -> np.ndarray:
    n = returns_matrix.shape[1]
    if _HAS_SKLEARN and returns_matrix.shape[0] > n:
        lw = LedoitWolf()
        lw.fit(returns_matrix)
        cov = lw.covariance_ * annual_factor
    else:
        cov = np.cov(returns_matrix.T) * annual_factor
        mu = np.trace(cov) / n
        shrinkage = min(_MAX_SHRINKAGE, n / max(returns_matrix.shape[0], 1))
        cov = (1 - shrinkage) * cov + shrinkage * mu * np.eye(n)

    cov += np.eye(n) * _COV_REGULARIZATION
    return cov

class WeightOptimizer:
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        annual_trading_days: int = 0,
        min_data_points: int = 0,
        scipy_method: str = '',
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        n_restarts: int = 5,
    ):
        opt_config = PORTFOLIO_CONFIG['optimization']
        self.risk_free_rate = risk_free_rate if risk_free_rate > 0 else opt_config['risk_free_rate']
        self.annual_trading_days = annual_trading_days if annual_trading_days > 0 else opt_config['annual_trading_days']
        self.min_data_points = min_data_points if min_data_points > 0 else opt_config['min_data_points']
        self.scipy_method = scipy_method if scipy_method else opt_config['scipy_method']
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.n_restarts = max(1, n_restarts)

    _REQUIRES_ANALYSIS = {'score', 'score_risk_adjusted', 'black_litterman'}
    _REQUIRES_RETURNS = {'score_risk_adjusted', 'markowitz', 'risk_parity', 'black_litterman'}

    def optimize(
        self,
        tickers: List[str],
        method: str = '',
        returns_data: pd.DataFrame = None,
        analysis_results: Dict = None
    ) -> Dict[str, float]:

        if not tickers:
            raise ValueError("La lista de tickers no puede estar vacía")

        if not method:
            method = PORTFOLIO_CONFIG['optimization']['default_method']

        strategies = {
            'equal': lambda: self._equal_weights(tickers),
            'score': lambda: self._score_weights(tickers, analysis_results),
            'score_risk_adjusted': lambda: self._score_risk_adjusted_weights(tickers, analysis_results, returns_data),
            'markowitz': lambda: self._markowitz_weights(tickers, returns_data),
            'risk_parity': lambda: self._risk_parity_weights(tickers, returns_data),
            'black_litterman': lambda: self._black_litterman_weights(tickers, returns_data, analysis_results),
        }

        if method not in strategies:
            return self._equal_weights(tickers)

        has_analysis = analysis_results is not None
        has_returns = returns_data is not None and not returns_data.empty

        if method in self._REQUIRES_ANALYSIS and not has_analysis:
            return self._equal_weights(tickers)
        if method in self._REQUIRES_RETURNS and not has_returns:
            return self._equal_weights(tickers)

        return strategies[method]()

    # ─────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────

    def _equal_weights(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            raise ValueError("La lista de tickers no puede estar vacía")
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    def _prepare_returns(
        self, tickers: List[str], returns: pd.DataFrame
    ) -> Optional[Tuple[List[str], pd.DataFrame, int, np.ndarray]]:
        """Filter tickers, validate data, compute covariance. Returns None on failure."""
        available = [t for t in tickers if t in returns.columns]
        if not available:
            return None
        r = returns[available].dropna()
        if len(r) < self.min_data_points:
            return None
        n = len(available)
        cov = _shrink_covariance(r.values, self.annual_trading_days)
        return available, r, n, cov

    def _multi_start_optimize(
        self,
        objective: Callable,
        n: int,
        bounds: tuple,
        initial_weights: np.ndarray,
        seed: int = 42,
        maxiter: int = 1000,
        ftol: float = 1e-9,
    ):
        """Run scipy minimize from multiple starting points, return best result."""
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        min_bound = bounds[0][0]

        rng = np.random.default_rng(seed)
        starting_points = [initial_weights]
        for _ in range(self.n_restarts):
            x = rng.dirichlet(np.ones(n))
            x = np.clip(x, min_bound, self.max_weight)
            x /= x.sum()
            starting_points.append(x)

        best = None
        for x0 in starting_points:
            res = minimize(
                objective, x0=x0, method=self.scipy_method,
                bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': False},
            )
            if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        return best

    def _build_weights(
        self,
        result,
        available_tickers: List[str],
        all_tickers: List[str],
        n: int,
        fallback_weights: np.ndarray = None,
    ) -> Optional[Dict[str, float]]:
        """Clip, normalize, and assemble weight dict from optimization result."""
        if result is None or not np.isfinite(result.fun):
            return None
        w = np.clip(result.x, 0.0, 1.0)
        default = fallback_weights if fallback_weights is not None else np.full(n, 1.0 / n)
        w = w / w.sum() if w.sum() > 0 else default
        weights = {t: float(wi) for t, wi in zip(available_tickers, w)}
        for t in all_tickers:
            if t not in weights:
                weights[t] = 0.0
        return weights

    # ─────────────────────────────────────────────
    # Simple methods
    # ─────────────────────────────────────────────

    def _score_weights(self, tickers: List[str], results: Dict) -> Dict[str, float]:
        raw = np.array([
            float(results.get(t, {}).get('scores', {}).get('total', 0.0))
            for t in tickers
        ], dtype=float)
        raw = np.maximum(raw, 0.0)
        s = raw.sum()
        if s <= 0:
            return self._equal_weights(tickers)
        w = raw / s
        return {t: float(wi) for t, wi in zip(tickers, w)}

    def _score_risk_adjusted_weights(
        self,
        tickers: List[str],
        results: Dict,
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        available_tickers = [t for t in tickers if t in returns.columns]
        if not available_tickers:
            return self._equal_weights(tickers)

        r = returns[available_tickers].dropna()
        if len(r) < self.min_data_points:
            return self._equal_weights(tickers)

        vol = r.std() * np.sqrt(self.annual_trading_days)
        vol = vol.replace(0, np.nan).fillna(vol.mean())
        if vol.isna().all():
            return self._equal_weights(tickers)

        raw_scores = np.array([
            float(results.get(t, {}).get('scores', {}).get('total', 0.0))
            for t in available_tickers
        ], dtype=float)
        raw_scores = np.maximum(raw_scores, 0.0)

        raw = raw_scores / vol.values
        s = raw.sum()
        if s <= 0:
            return self._equal_weights(available_tickers)

        w = raw / s
        weights = {t: float(wi) for t, wi in zip(available_tickers, w)}
        for t in tickers:
            if t not in weights:
                weights[t] = 0.0
        return weights

    # ─────────────────────────────────────────────
    # Markowitz (Maximum Sharpe) with Ledoit-Wolf + multiple restarts
    # ─────────────────────────────────────────────

    def _markowitz_weights(self, tickers: List[str], returns: pd.DataFrame) -> Dict[str, float]:
        try:
            prep = self._prepare_returns(tickers, returns)
            if prep is None:
                return self._equal_weights(tickers)
            available, r, n, cov = prep

            mean_ret = r.mean().values * self.annual_trading_days

            def neg_sharpe(w):
                ret = np.dot(w, mean_ret)
                vol = np.sqrt(w @ cov @ w)
                return -(ret - self.risk_free_rate) / vol if vol > _VOL_FLOOR else _PENALTY

            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
            result = self._multi_start_optimize(
                neg_sharpe, n, bounds,
                initial_weights=np.full(n, 1.0 / n), seed=42,
            )
            weights = self._build_weights(result, available, tickers, n)
            return weights if weights is not None else self._equal_weights(tickers)

        except Exception:
            return self._equal_weights(tickers)

    # ─────────────────────────────────────────────
    # Risk Parity (Equal Risk Contribution)
    # ─────────────────────────────────────────────

    def _risk_parity_weights(self, tickers: List[str], returns: pd.DataFrame) -> Dict[str, float]:
        try:
            prep = self._prepare_returns(tickers, returns)
            if prep is None:
                return self._equal_weights(tickers)
            available, r, n, cov = prep

            def risk_parity_obj(w):
                port_var = w @ cov @ w
                mrc = cov @ w / np.sqrt(port_var)
                rc = w * mrc
                target = np.sqrt(port_var) / n
                return np.sum((rc - target) ** 2)

            lower = max(self.min_weight, _RISK_PARITY_MIN_BOUND)
            bounds = tuple((lower, self.max_weight) for _ in range(n))
            result = self._multi_start_optimize(
                risk_parity_obj, n, bounds,
                initial_weights=np.full(n, 1.0 / n), seed=0,
                maxiter=2000, ftol=1e-12,
            )
            weights = self._build_weights(result, available, tickers, n)
            return weights if weights is not None else self._equal_weights(tickers)

        except Exception:
            return self._equal_weights(tickers)

    # ─────────────────────────────────────────────
    # Black-Litterman with numerical stability improvements
    # ─────────────────────────────────────────────

    def _black_litterman_weights(
        self,
        tickers: List[str],
        returns: pd.DataFrame,
        analysis_results: Dict
    ) -> Dict[str, float]:
        try:
            prep = self._prepare_returns(tickers, returns)
            if prep is None:
                return self._equal_weights(tickers)
            available, r, n, cov = prep

            w_mkt = np.ones(n) / n

            market_return = (r.mean().values * self.annual_trading_days).mean()
            market_variance = float(w_mkt @ cov @ w_mkt)
            delta = (market_return - self.risk_free_rate) / max(market_variance, _VOL_FLOOR)
            delta = np.clip(delta, _BL_DELTA_MIN, _BL_DELTA_MAX)

            pi = delta * (cov @ w_mkt)

            tau = np.clip(1.0 / len(r), _BL_TAU_MIN, _BL_TAU_MAX)

            P = np.eye(n)

            scores = np.array([
                float(analysis_results.get(t, {}).get('scores', {}).get('total', _BL_NEUTRAL_SCORE))
                for t in available
            ])
            scores = np.clip(scores, 0.0, 100.0)

            score_deviations = (scores - _BL_NEUTRAL_SCORE) / _BL_NEUTRAL_SCORE
            Q = pi + score_deviations * _BL_MAX_VIEW_RETURN

            confidence = np.abs(scores - _BL_NEUTRAL_SCORE) / _BL_NEUTRAL_SCORE
            confidence = np.clip(confidence, _BL_MIN_CONFIDENCE, 1.0)
            base_uncertainty = np.diag(tau * (P @ cov @ P.T))
            omega = np.diag(base_uncertainty / confidence)

            tau_cov = tau * cov
            cond = np.linalg.cond(tau_cov)
            if cond > _BL_COND_THRESHOLD:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "BL: ill-conditioned τΣ matrix (cond=%.1e), falling back to Markowitz", cond
                )
                return self._markowitz_weights(tickers, returns)

            tau_cov_inv_pi = np.linalg.solve(tau_cov, pi)
            omega_inv_Q = Q / np.diag(omega)
            omega_inv_diag = 1.0 / np.diag(omega)
            tau_cov_inv = np.linalg.solve(tau_cov, np.eye(n))
            precision = tau_cov_inv + np.diag(omega_inv_diag)
            rhs = tau_cov_inv_pi + omega_inv_Q
            mu_bl = np.linalg.solve(precision, rhs)

            def neg_sharpe_bl(w):
                port_return = np.dot(w, mu_bl)
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_return - self.risk_free_rate) / port_vol if port_vol > _VOL_FLOOR else _PENALTY

            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
            result = self._multi_start_optimize(
                neg_sharpe_bl, n, bounds,
                initial_weights=w_mkt.copy(), seed=42,
            )
            weights = self._build_weights(result, available, tickers, n, fallback_weights=w_mkt)
            if weights is not None:
                return weights
            return {t: float(wi) for t, wi in zip(available, w_mkt)}

        except Exception:
            return self._equal_weights(tickers)
