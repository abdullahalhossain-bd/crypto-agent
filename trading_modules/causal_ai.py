"""
Causal AI Module — Cause vs Correlation
========================================

Answers "WHY does this signal work?" vs "THIS signal correlates with returns."

Correlation: NFP ↑ → Gold ↑ (they move together)
Causal:     NFP ↑ CAUSES USD ↑ CAUSES Gold ↓ (mechanism understood)

Two methods:
  1. Double Machine Learning (DML) — isolates treatment effect from confounders
  2. Granger Causality — does X help predict Y beyond Y's own history?

Source: ml4t-3e (review #18) ch.15 — Causal Machine Learning
        Vibe-Trading crypto_perps_funding case study (treatment=premium_zscore)

Usage:
    from trading_modules.causal_ai import DoubleML, GrangerCausality

    # DML: Is funding rate causally affecting returns?
    dml = DoubleML()
    result = dml.estimate_effect(
        treatment=funding_rates,      # What we're testing
        outcome=forward_returns,       # What we want to predict
        confounders=features_df,       # Variables that affect both
    )
    # → effect=0.002, p_value=0.03, significant=True

    # Granger: Does BTC volume Granger-cause ETH returns?
    gc = GrangerCausality(max_lag=10)
    result = gc.test(btc_volume, eth_returns)
    # → p_value=0.01, granger_causes=True
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from statistics import NormalDist


# ═══════════════════════════════════════════════════════════════
# 1. Double Machine Learning
# ═══════════════════════════════════════════════════════════════

@dataclass
class CausalEffect:
    """Estimated causal effect."""
    effect: float          # Average Treatment Effect (ATE)
    std_error: float       # Standard error
    t_statistic: float
    p_value: float
    is_significant: bool   # p < 0.05
    confidence_interval: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "effect": round(self.effect, 6),
            "std_error": round(self.std_error, 6),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "significant": self.is_significant,
            "ci_95": [round(self.confidence_interval[0], 6),
                      round(self.confidence_interval[1], 6)],
        }


class DoubleML:
    """
    Double Machine Learning for causal inference.

    Separates the causal effect of a treatment variable from
    confounding factors using a "double debiasing" approach:

    1. Train ML model to predict treatment from confounders → residuals (T_resid)
    2. Train ML model to predict outcome from confounders → residuals (Y_resid)
    3. Regress Y_resid on T_resid → causal effect estimate

    This removes the bias from confounders that affect both treatment and outcome.

    Reference: Chernozhukov et al. (2018) "double/debiased machine learning"
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: "random_forest", "xgboost", or "linear"
        """
        self.model_type = model_type

    def estimate_effect(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        confounders: pd.DataFrame,
    ) -> CausalEffect:
        """
        Estimate the causal effect of treatment on outcome.

        Args:
            treatment: Treatment variable (e.g., funding rate)
            outcome: Outcome variable (e.g., forward returns)
            confounders: DataFrame of confounding variables

        Returns:
            CausalEffect with ATE, p-value, and confidence interval
        """
        from sklearn.ensemble import RandomForestRegressor

        X = confounders.values
        T = np.array(treatment).ravel()
        Y = np.array(outcome).ravel()

        # Align lengths
        min_len = min(len(X), len(T), len(Y))
        X, T, Y = X[:min_len], T[:min_len], Y[:min_len]

        # Remove NaN
        mask = ~(np.isnan(T) | np.isnan(Y) | np.any(np.isnan(X), axis=1))
        X, T, Y = X[mask], T[mask], Y[mask]

        if len(X) < 50:
            return CausalEffect(0, 0, 0, 1.0, False, (0, 0))

        # Step 1: Predict treatment from confounders
        model_T = self._create_model()
        model_T.fit(X, T)
        T_pred = model_T.predict(X)
        T_resid = T - T_pred

        # Step 2: Predict outcome from confounders
        model_Y = self._create_model()
        model_Y.fit(X, Y)
        Y_pred = model_Y.predict(X)
        Y_resid = Y - Y_pred

        # Step 3: Regress Y_resid on T_resid
        # OLS: beta = (T_resid' T_resid)^{-1} T_resid' Y_resid
        numerator = np.sum(T_resid * Y_resid)
        denominator = np.sum(T_resid ** 2)

        if abs(denominator) < 1e-10:
            return CausalEffect(0, 0, 0, 1.0, False, (0, 0))

        ate = numerator / denominator

        # Standard error
        residuals = Y_resid - ate * T_resid
        n = len(T_resid)
        se = np.sqrt(np.sum(residuals ** 2) / (n - 1) / denominator)

        # T-statistic and p-value
        t_stat = ate / se if se > 1e-10 else 0
        p_value = 2 * (1 - NormalDist().cdf(abs(t_stat)))

        # 95% CI
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return CausalEffect(
            effect=float(ate),
            std_error=float(se),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < 0.05,
            confidence_interval=(float(ci_lower), float(ci_upper)),
        )

    def _create_model(self):
        """Create nuisance model."""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        else:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()


# ═══════════════════════════════════════════════════════════════
# 2. Granger Causality
# ═══════════════════════════════════════════════════════════════

@dataclass
class GrangerResult:
    """Granger causality test result."""
    causes: bool
    p_value: float
    best_lag: int
    f_statistic: float

    def to_dict(self) -> dict:
        return {
            "causes": self.causes,
            "p_value": round(self.p_value, 6),
            "best_lag": self.best_lag,
            "f_statistic": round(self.f_statistic, 4),
        }


class GrangerCausality:
    """
    Granger causality test.

    Tests whether time series X Granger-causes time series Y:
    Does past values of X help predict Y beyond Y's own history?

    Method: Compare restricted model (Y ~ Y_lags) vs unrestricted (Y ~ Y_lags + X_lags)
    If unrestricted is significantly better → X Granger-causes Y

    Note: Granger causality is NOT true causality — it's predictive causality.
    """

    def __init__(self, max_lag: int = 10, significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance

    def test(self, x: np.ndarray, y: np.ndarray) -> GrangerResult:
        """
        Test if X Granger-causes Y.

        Args:
            x: Potential cause time series
            y: Effect time series

        Returns:
            GrangerResult
        """
        from scipy import stats

        x = np.array(x, dtype=float).ravel()
        y = np.array(y, dtype=float).ravel()

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < self.max_lag + 20:
            return GrangerResult(causes=False, p_value=1.0, best_lag=0, f_statistic=0)

        best_p = 1.0
        best_lag = 0
        best_f = 0.0

        for lag in range(1, min(self.max_lag + 1, len(x) - 20)):
            # Build lagged features
            y_lagged = np.column_stack([y[i:len(y)-lag+i] for i in range(lag)])
            x_lagged = np.column_stack([x[i:len(x)-lag+i] for i in range(lag)])
            y_target = y[lag:]

            # Restricted model: Y ~ Y_lags
            X_restricted = np.column_stack([y_lagged, np.ones(len(y_target))])
            beta_r = np.linalg.lstsq(X_restricted, y_target, rcond=None)[0]
            resid_r = y_target - X_restricted @ beta_r
            ssr_r = np.sum(resid_r ** 2)

            # Unrestricted model: Y ~ Y_lags + X_lags
            X_unrestricted = np.column_stack([y_lagged, x_lagged, np.ones(len(y_target))])
            beta_u = np.linalg.lstsq(X_unrestricted, y_target, rcond=None)[0]
            resid_u = y_target - X_unrestricted @ beta_u
            ssr_u = np.sum(resid_u ** 2)

            # F-test
            n = len(y_target)
            p_r = X_restricted.shape[1]
            p_u = X_unrestricted.shape[1]
            df1 = p_u - p_r
            df2 = n - p_u

            if df2 <= 0 or ssr_u <= 0:
                continue

            f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)

            if p_value < best_p:
                best_p = p_value
                best_lag = lag
                best_f = f_stat

        return GrangerResult(
            causes=best_p < self.significance,
            p_value=float(best_p),
            best_lag=best_lag,
            f_statistic=float(best_f),
        )


# ═══════════════════════════════════════════════════════════════
# 3. Causal Summary
# ═══════════════════════════════════════════════════════════════

def causal_summary(
    treatment_name: str,
    outcome_name: str,
    effect: CausalEffect,
    granger: Optional[GrangerResult] = None,
) -> str:
    """
    Generate human-readable causal analysis summary.

    Usage:
        summary = causal_summary("funding_rate", "forward_return", dml_result, gc_result)
        print(summary)
    """
    lines = [
        f"## Causal Analysis: {treatment_name} → {outcome_name}",
        "",
        f"### Double ML Estimate",
        f"- Average Treatment Effect: {effect.effect:.6f}",
        f"- Standard Error: {effect.std_error:.6f}",
        f"- T-statistic: {effect.t_statistic:.4f}",
        f"- P-value: {effect.p_value:.4f}",
        f"- 95% CI: [{effect.confidence_interval[0]:.6f}, {effect.confidence_interval[1]:.6f}]",
        f"- Significant: {'✅ Yes' if effect.is_significant else '❌ No'}",
    ]

    if granger:
        lines.extend([
            "",
            f"### Granger Causality",
            f"- Best Lag: {granger.best_lag}",
            f"- F-statistic: {granger.f_statistic:.4f}",
            f"- P-value: {granger.p_value:.4f}",
            f"- Granger-causes: {'✅ Yes' if granger.causes else '❌ No'}",
        ])

    # Interpretation
    lines.append("")
    if effect.is_significant and (granger is None or granger.causes):
        direction = "positive" if effect.effect > 0 else "negative"
        lines.append(f"### Interpretation")
        lines.append(f"{treatment_name} has a **{direction} causal effect** on {outcome_name}.")
        lines.append(f"A unit increase in {treatment_name} causes {outcome_name} to change by {effect.effect:.6f}.")
    elif effect.is_significant:
        lines.append(f"### Interpretation")
        lines.append(f"DML finds a significant effect but Granger test does not confirm.")
        lines.append(f"The relationship may be confounded or non-linear.")
    else:
        lines.append(f"### Interpretation")
        lines.append(f"No significant causal effect detected.")
        lines.append(f"The observed correlation is likely due to confounders.")

    return "\n".join(lines)
