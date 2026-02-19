import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.optimize import minimize
from ....tools.config import PORTFOLIO_CONFIG

# Ledoit-Wolf shrinkage: best-in-class covariance estimator for small samples
try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _shrink_covariance(returns_matrix: np.ndarray, annual_factor: int) -> np.ndarray:
    """
    Estimate annualised covariance matrix with Ledoit-Wolf shrinkage.
    Falls back to sample covariance + diagonal regularisation if sklearn is unavailable.
    """
    n = returns_matrix.shape[1]
    if _HAS_SKLEARN and returns_matrix.shape[0] > n:
        lw = LedoitWolf()
        lw.fit(returns_matrix)
        cov = lw.covariance_ * annual_factor
    else:
        cov = np.cov(returns_matrix.T) * annual_factor
        # Manual Ledoit-Wolf-style shrinkage toward scaled identity
        mu = np.trace(cov) / n
        shrinkage = min(0.1, n / max(returns_matrix.shape[0], 1))
        cov = (1 - shrinkage) * cov + shrinkage * mu * np.eye(n)

    # Numerical floor to guarantee positive-definiteness
    cov += np.eye(n) * 1e-8
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
        # min_weight: 0.0 = allow zero; e.g. 0.02 enforces at least 2% per asset
        self.min_weight = min_weight
        # n_restarts: number of random starting points tried in numerical optimisers
        self.n_restarts = max(1, n_restarts)

    # ─────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────

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
    # Simple methods
    # ─────────────────────────────────────────────

    def _equal_weights(self, tickers: List[str]) -> Dict[str, float]:
        if not tickers:
            raise ValueError("La lista de tickers no puede estar vacía")
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

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
            available_tickers = [t for t in tickers if t in returns.columns]
            if not available_tickers:
                return self._equal_weights(tickers)

            r = returns[available_tickers].dropna()
            if len(r) < self.min_data_points:
                return self._equal_weights(tickers)

            n = len(available_tickers)
            mean_ret = r.mean().values * self.annual_trading_days
            cov = _shrink_covariance(r.values, self.annual_trading_days)

            def neg_sharpe(w):
                ret = np.dot(w, mean_ret)
                vol = np.sqrt(w @ cov @ w)
                return -(ret - self.risk_free_rate) / vol if vol > 1e-10 else 1e10

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))

            best_result = None
            rng = np.random.default_rng(42)

            # Equal-weight start + n_restarts random starts
            starting_points = [np.full(n, 1.0 / n)]
            for _ in range(self.n_restarts):
                x = rng.dirichlet(np.ones(n))
                x = np.clip(x, self.min_weight, self.max_weight)
                x /= x.sum()
                starting_points.append(x)

            for x0 in starting_points:
                res = minimize(
                    neg_sharpe,
                    x0=x0,
                    method=self.scipy_method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
                )
                # Accept even if result.success=False when the value is finite and better
                if np.isfinite(res.fun) and (best_result is None or res.fun < best_result.fun):
                    best_result = res

            if best_result is not None and np.isfinite(best_result.fun):
                w = np.clip(best_result.x, 0.0, 1.0)
                w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
                weights = {t: float(wi) for t, wi in zip(available_tickers, w)}
                for t in tickers:
                    if t not in weights:
                        weights[t] = 0.0
                return weights

            return self._equal_weights(tickers)

        except Exception:
            return self._equal_weights(tickers)

    # ─────────────────────────────────────────────
    # Risk Parity (Equal Risk Contribution)
    # ─────────────────────────────────────────────

    def _risk_parity_weights(self, tickers: List[str], returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk Parity / Equal Risk Contribution (ERC).
        Each asset contributes the same fraction of total portfolio risk.
        Robust: does not require expected returns, only the covariance matrix.
        """
        try:
            available_tickers = [t for t in tickers if t in returns.columns]
            if not available_tickers:
                return self._equal_weights(tickers)

            r = returns[available_tickers].dropna()
            if len(r) < self.min_data_points:
                return self._equal_weights(tickers)

            n = len(available_tickers)
            cov = _shrink_covariance(r.values, self.annual_trading_days)

            # Objective: minimise sum of squared deviations from equal risk contribution
            def risk_parity_obj(w):
                port_var = w @ cov @ w
                # Marginal risk contributions: ∂σ/∂w_i = (Σw)_i / σ
                mrc = cov @ w / np.sqrt(port_var)
                # Risk contributions: RC_i = w_i * mrc_i
                rc = w * mrc
                # Target: RC_i = port_vol / n for all i
                target = np.sqrt(port_var) / n
                return np.sum((rc - target) ** 2)

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((max(self.min_weight, 1e-4), self.max_weight) for _ in range(n))

            best_result = None
            rng = np.random.default_rng(0)
            starting_points = [np.full(n, 1.0 / n)]
            for _ in range(self.n_restarts):
                x = rng.dirichlet(np.ones(n))
                x = np.clip(x, 1e-4, self.max_weight)
                x /= x.sum()
                starting_points.append(x)

            for x0 in starting_points:
                res = minimize(
                    risk_parity_obj,
                    x0=x0,
                    method=self.scipy_method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
                )
                if np.isfinite(res.fun) and (best_result is None or res.fun < best_result.fun):
                    best_result = res

            if best_result is not None and np.isfinite(best_result.fun):
                w = np.clip(best_result.x, 0.0, 1.0)
                w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
                weights = {t: float(wi) for t, wi in zip(available_tickers, w)}
                for t in tickers:
                    if t not in weights:
                        weights[t] = 0.0
                return weights

            return self._equal_weights(tickers)

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
            available_tickers = [t for t in tickers if t in returns.columns]
            if not available_tickers:
                return self._equal_weights(tickers)

            r = returns[available_tickers].dropna()
            if len(r) < self.min_data_points:
                return self._equal_weights(tickers)

            n = len(available_tickers)
            # Ledoit-Wolf covariance (already annualised)
            cov = _shrink_covariance(r.values, self.annual_trading_days)

            # Market equilibrium: equal-weight proxy (no market caps available)
            w_mkt = np.ones(n) / n

            market_return = (r.mean().values * self.annual_trading_days).mean()
            market_variance = float(w_mkt @ cov @ w_mkt)
            delta = (market_return - self.risk_free_rate) / max(market_variance, 1e-10)
            delta = np.clip(delta, 2.0, 4.0)

            # Implied equilibrium returns
            pi = delta * (cov @ w_mkt)

            # Uncertainty in the prior: τ ∈ [0.01, 0.1]
            tau = np.clip(1.0 / len(r), 0.01, 0.1)

            # Views: absolute, one per asset (P = I)
            P = np.eye(n)

            scores = np.array([
                float(analysis_results.get(t, {}).get('scores', {}).get('total', 50.0))
                for t in available_tickers
            ])
            scores = np.clip(scores, 0.0, 100.0)

            neutral_score = 50.0
            max_view_return = 0.15
            score_deviations = (scores - neutral_score) / 50.0
            Q = pi + score_deviations * max_view_return

            # View uncertainty: less confident when score is close to neutral
            confidence = np.abs(scores - neutral_score) / 50.0
            confidence = np.clip(confidence, 0.1, 1.0)
            base_uncertainty = np.diag(tau * (P @ cov @ P.T))
            omega = np.diag(base_uncertainty / confidence)

            # ── Posterior mean via linear system (avoids direct matrix inversion) ──
            # BL formula: μ_BL = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q]
            tau_cov = tau * cov

            # Condition number check — fall back to Markowitz if ill-conditioned
            cond = np.linalg.cond(tau_cov)
            if cond > 1e12:
                import logging as _logging
                _logging.getLogger(__name__).warning("BL: ill-conditioned τΣ matrix (cond=%.1e), falling back to Markowitz", cond)
                return self._markowitz_weights(tickers, returns)

            # Solve linear systems instead of explicit inversions
            # (τΣ)⁻¹ π  →  solve (τΣ) x = π
            tau_cov_inv_pi = np.linalg.solve(tau_cov, pi)

            # Ω⁻¹ Q  →  solve Ω x = Q  (Ω is diagonal, so trivial)
            omega_inv_Q = Q / np.diag(omega)

            # Pᵀ Ω⁻¹ P  (P = I, so this is just Ω⁻¹ as a diagonal matrix)
            omega_inv_diag = 1.0 / np.diag(omega)

            # Precision matrix: M = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
            # (τΣ)⁻¹  →  solve (τΣ) X = I
            tau_cov_inv = np.linalg.solve(tau_cov, np.eye(n))
            precision = tau_cov_inv + np.diag(omega_inv_diag)

            # Posterior covariance: cov_post = precision⁻¹
            # Posterior mean: μ_BL = cov_post @ (tau_cov_inv_pi + Pᵀ Ω⁻¹ Q)
            rhs = tau_cov_inv_pi + omega_inv_Q      # P = I so Pᵀ = I
            mu_bl = np.linalg.solve(precision, rhs)

            # ── Optimise Sharpe with BL expected returns ──
            def neg_sharpe_bl(w):
                port_return = np.dot(w, mu_bl)
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_return - self.risk_free_rate) / port_vol if port_vol > 1e-10 else 1e10

            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))

            best_result = None
            rng = np.random.default_rng(42)
            starting_points = [w_mkt.copy()]
            for _ in range(self.n_restarts):
                x = rng.dirichlet(np.ones(n))
                x = np.clip(x, self.min_weight, self.max_weight)
                x /= x.sum()
                starting_points.append(x)

            for x0 in starting_points:
                res = minimize(
                    neg_sharpe_bl,
                    x0=x0,
                    method=self.scipy_method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
                )
                if np.isfinite(res.fun) and (best_result is None or res.fun < best_result.fun):
                    best_result = res

            if best_result is not None and np.isfinite(best_result.fun):
                w = np.clip(best_result.x, 0.0, 1.0)
                w = w / w.sum() if w.sum() > 0 else w_mkt
                weights = {t: float(wi) for t, wi in zip(available_tickers, w)}
                for t in tickers:
                    if t not in weights:
                        weights[t] = 0.0
                return weights

            return {t: float(wi) for t, wi in zip(available_tickers, w_mkt)}

        except Exception:
            return self._equal_weights(tickers)
