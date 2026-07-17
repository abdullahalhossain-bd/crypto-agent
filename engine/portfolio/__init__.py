"""engine.portfolio package — portfolio-level exposure, correlation, allocation.

NOTE: The canonical PortfolioManager now lives at
architecture/portfolio_manager_v2.py. The old engine/portfolio/portfolio_manager.py
was archived to legacy/ in Phase 3 (it was a Stack-B duplicate, never on the
canonical Stack-C path). This package now only exposes the exposure and
correlation helpers that are still in use.
"""
from engine.portfolio.exposure_model import ExposureModel, Exposure  # noqa: F401
from engine.portfolio.correlation_matrix import CorrelationMatrix  # noqa: F401

__all__ = [
    "ExposureModel",
    "Exposure",
    "CorrelationMatrix",
]
